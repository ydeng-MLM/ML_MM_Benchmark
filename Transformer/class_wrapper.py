"""
The class wrapper for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler
# Libs
import numpy as np
from math import inf
import pandas as pd
# Own module
from utils.time_recorder import time_keeper

class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')    # Set the BVL to large number
        self.best_training_loss = float('inf')    # Set the BTL to large number

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        # encoder = Encoder(self.flags)
        # decoder = Decoder(self.flags)
        # spec_enc = SpectraEncoder(self.flags)
        model = self.model_fn(self.flags)
        print(model)
        return model

    def make_loss(self, logit=None, labels=None, G=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network, the predicted geometry
        :param labels: The ground truth labels, the Truth geometry
        :param boundary: The boolean flag indicating adding boundary loss or not
        :param z_log_var: The log variance for VAE kl_loss
        :param z_mean: The z mean vector for VAE kl_loss
        :return: the total loss
        """
        MSE_loss = nn.functional.mse_loss(logit, labels, reduction='mean')          # The MSE Loss
        BDY_loss = 0
        if G is not None:         # This is using the boundary loss
            X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
            X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            relu = torch.nn.ReLU()
            BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)

    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op
    
    def make_optimizer_eval(self, geometry_eval, optimizer_type=None):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def build_tensor(self, nparray, requires_grad=False):
        return torch.tensor(nparray, requires_grad=requires_grad, device='cuda', dtype=torch.float)

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        if self.flags.lr_scheduler == 'warm_restart':
            return lr_scheduler.CosineAnnealingWarmRestarts(optm, self.flags.warm_restart_T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False) 
        elif self.flags.lr_scheduler == 'reduce_plateau':
            return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
        self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
    
    def get_cuda_memory_info(self):
        r = torch.cuda.memory_reserved(0)/1e9
        a = torch.cuda.memory_allocated(0)/1e9
        print('after delete, reserved memory {}G, allocated memory {}G'.format(r,a))

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        print("Starting training now")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                S_pred = self.model(geometry)
                loss = self.make_loss(logit=S_pred, labels=spectra)
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss.cpu().data.numpy()                                  # Aggregate the loss
                del S_pred, loss
                ################ Debugging the memory cuda #################
                #print('in epoch {} test batch {}, lenghth is :'.format(epoch, j))
                #self.get_cuda_memory_info()
                ################ Debugging the memory cuda #################

            # Calculate the avg loss of training
            train_avg_loss = train_loss / (j + 1)
            # Recording the best training loss
            if train_avg_loss < self.best_training_loss:
                self.best_training_loss = train_avg_loss

            if epoch % self.flags.eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/total_train', train_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    ################ Debugging the memory cuda #################
                    #print('in epoch {} test batch {}, lenghth is :'.format(epoch, j))
                    #self.get_cuda_memory_info()
                    ################ Debugging the memory cuda #################


                    S_pred = self.model(geometry)
                    loss = self.make_loss(logit=S_pred, labels=spectra)
                    #print(loss.cpu().data.numpy())
                    test_loss += loss.cpu().data.numpy()                                       # Aggregate the loss
                    del loss, S_pred
                    ################ Debugging the memory cuda #################
                    #self.get_cuda_memory_info()
                    ################ Debugging the memory cuda #################

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss/ (j+1)
                self.log.add_scalar('Loss/total_test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        tk.record(1)                # Record the total time of the training peroid

    def evaluate(self, save_dir='data/', prefix=''):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        # Set to evaluation mode for batch_norm layers
        self.model.eval()
        
        saved_model_str = self.saved_model.replace('/','_') + prefix
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))

        tk = time_keeper(os.path.join(save_dir, 'evaluation_time.txt'))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                Ypred = self.model(geometry).cpu().data.numpy()
                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                np.savetxt(fyp, Ypred)
        tk.record(1)                # Record the total time of the eval period
        return Ypred_file, Ytruth_file
    

    def evaluate_multiple_time(self, time=200, save_dir='/home/sr365/mm_bench_multi_eval/VAE/'):
        """
        Make evaluation multiple time for deeper comparison for stochastic algorithms
        :param save_dir: The directory to save the result
        :return:
        """
        save_dir = os.path.join(save_dir, self.flags.data_set)
        tk = time_keeper(os.path.join(save_dir, 'evaluation_time.txt'))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for i in range(time):
            self.evaluate(save_dir=save_dir, prefix='inference' + str(i))
            tk.record(i)


    def evaluate_inverse(self, save_dir='data/', save_all=False, MSE_Simulator=False, save_misc=False, save_Simulator_Ypred=True):
        """
        The function to evaluate how good the Neural Adjoint is and output results
        :param save_dir: The directory to save the results
        :param save_all: Save all the results instead of the best one (T_200 is the top 200 ones)
        :param MSE_Simulator: Use simulator loss to sort (DO NOT ENABLE THIS, THIS IS OK ONLY IF YOUR APPLICATION IS FAST VERIFYING)
        :param save_misc: save all the details that are probably useless
        :param save_Simulator_Ypred: Save the Ypred that the Simulator gives
        (This is useful as it gives us the true Ypred instead of the Ypred that the network "thinks" it gets, which is
        usually inaccurate due to forward model error)
        :return:
        """
        self.load()                             # load the model as constructed
        try:
            bs = self.flags.backprop_step         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.backprop_step = 300
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        print("evalution output pattern:", Ypred_file)

        # Time keeping
        #tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation_time.txt'))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                Xpred, Ypred, loss = self.evaluate_one(spectra, save_dir=save_dir, save_all=save_all, ind=ind,
                                                        MSE_Simulator=MSE_Simulator, save_misc=save_misc, save_Simulator_Ypred=save_Simulator_Ypred)
                #tk.record(ind)                          # Keep the time after each evaluation for backprop
                # self.plot_histogram(loss, ind)                                # Debugging purposes
                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                if self.flags.data_set != 'meta_material':
                    np.savetxt(fyp, Ypred)
                np.savetxt(fxp, Xpred)
                if ind > 10:
                    break
        return Ypred_file, Ytruth_file
    
    def evaluate_one(self, target_spectra, save_dir='data/', MSE_Simulator=False ,save_all=False, ind=None, save_misc=False, save_Simulator_Ypred=False):
        """
        The function which being called during evaluation and evaluates one target y using # different trails
        :param target_spectra: The target spectra/y to backprop to 
        :param save_dir: The directory to save to when save_all flag is true
        :param MSE_Simulator: Use Simulator Loss to get the best instead of the default NN output logit
        :param save_all: The multi_evaluation where each trail is monitored (instad of the best) during backpropagation
        :param ind: The index of this target_spectra in the batch
        :param save_misc: The flag to print misc information for degbugging purposes, usually printed to best_mse
        :return: Xpred_best: The 1 single best Xpred corresponds to the best Ypred that is being backproped 
        :return: Ypred_best: The 1 singe best Ypred that is reached by backprop
        :return: MSE_list: The list of MSE at the last stage
        """

        # Initialize the geometry_eval or the initial guess xs
        geometry_eval = self.initialize_geometry_eval()
        # Set up the learning schedule and optimizer
        self.optm_eval = self.make_optimizer_eval(geometry_eval)#, optimizer_type='SGD')
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        
        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.eval_batch_size, -1])

        # Begin NA
        for i in range(self.flags.backprop_step):
            # Make the initialization from [-1, 1], can only be in loop due to gradient calculator constraint
            geometry_eval_input = self.initialize_from_uniform_to_dataset_distrib(geometry_eval)
            self.optm_eval.zero_grad()                                  # Zero the gradient first
            logit = self.model(geometry_eval_input)                     # Get the output
            ###################################################
            # Boundar loss controled here: with Boundary Loss #
            ###################################################
            loss = self.make_loss(logit, target_spectra_expand, G=geometry_eval_input)         # Get the loss
            loss.backward()                                             # Calculate the Gradient
            # update weights and learning rate scheduler
            if i != self.flags.backprop_step - 1:
                self.optm_eval.step()  # Move one step the optimizer
                self.lr_scheduler.step(loss.data)
        
        ###################################
        # From candidates choose the best #
        ###################################
        Ypred = logit.cpu().data.numpy()

        # calculate the MSE list and get the best one
        MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
        best_estimate_index = np.argmin(MSE_list)
        print("The best performing one is:", best_estimate_index)
        Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
        Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

        return Xpred_best, Ypred_best, MSE_list

    def initialize_geometry_eval(self):
        """
        Initialize the geometry eval according to different dataset. These 2 need different handling
        :return: The initialized geometry eval
        """
        geomtry_eval = torch.rand([self.flags.eval_batch_size, self.flags.dim_G], requires_grad=True, device='cuda')
        #geomtry_eval = torch.randn([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
        return geomtry_eval

    def initialize_from_uniform_to_dataset_distrib(self, geometry_eval):
        """
        since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
        to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
        of the X prior given in the training set and data generation process
        :param geometry_eval: The input uniform distribution from [0,1]
        :return: The transformed initial guess from prior distribution
        """
        X_range, X_lower_bound, X_upper_bound = self.get_boundary_lower_bound_uper_bound()
        geometry_eval_input = geometry_eval * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
        return geometry_eval_input
    
    def get_boundary_lower_bound_uper_bound(self):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        if self.flags.data_set == 'Peurifoy': 
            x_dim = 8
            return np.array([2 for i in range(x_dim)]), np.array([-1 for i in range(x_dim)]), np.array([1 for i in range(x_dim)])
        elif self.flags.data_set == 'Color': 
            x_dim = 3
            return np.array([1 for i in range(x_dim)]), np.array([0 for i in range(x_dim)]), np.array([1 for i in range(x_dim)])
        elif self.flags.data_set == 'Yang': 
            x_max = np.array([6.000000238418579102e-01, 1.500000000000000000e+00,2.000000029802322388e-01, 2.000000029802322388e-01
                    ,2.000000029802322388e-01,2.000000029802322388e-01,2.000000029802322388e-01,2.000000029802322388e-01
                    ,2.000000029802322388e-01,2.000000029802322388e-01,7.853999733924863502e-01,7.853999733924863502e-01
                    ,7.853999733924863502e-01,7.853999733924863502e-01])
            x_min = np.array([3.000000119209288996e-01,1.000000000000000000e+00,1.000000014901161471e-01,1.000000014901161471e-01
                    ,1.000000014901161471e-01,1.000000014901161471e-01,1.000000014901161471e-01,1.000000014901161471e-01
                    ,1.000000014901161471e-01,1.000000014901161471e-01,-7.853999733924863502e-01,-7.853999733924863502e-01
                    ,-7.853999733924863502e-01,-7.853999733924863502e-01])
            return x_max - x_min, x_min, x_max

    def predict(self, Ytruth_file, save_dir='data/', prefix=''):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/', '_') + prefix

        Ytruth = pd.read_csv(Ytruth_file, header=None, delimiter=',')     # Read the input
        if len(Ytruth.columns) == 1: # The file is not delimitered by ',' but ' '
            Ytruth = pd.read_csv(Ytruth_file, header=None, delimiter=' ')
        Ytruth_tensor = torch.from_numpy(Ytruth.values).to(torch.float)
        print('shape of Ytruth tensor :', Ytruth_tensor.shape)

        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        # keep time
        tk = time_keeper(os.path.join(save_dir, 'evaluation_time.txt'))
    
        if cuda:
            Ytruth_tensor = Ytruth_tensor.cuda()
        print('model in eval:', self.model)
        Xpred = self.model.inference(Ytruth_tensor).cpu().data.numpy()

        # Open those files to append
        with open(Ytruth_file, 'a') as fyt, open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            np.savetxt(fyt, Ytruth_tensor.cpu().data.numpy())
            np.savetxt(fxp, Xpred)
        tk.record(1)
        return Ypred_file, Ytruth_file
