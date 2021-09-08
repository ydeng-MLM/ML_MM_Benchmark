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
from model_maker import Transformer
from utils.evaluation_helper import plotMSELossDistrib

class Network(object):
    def __init__(self, dim_g, dim_s, feature_channel_num=32, nhead_encoder=8, 
                dim_fc_encoder=64,num_encoder_layer=6, head_linear=None, 
                tail_linear=None, sequence_length=8, model_name=None, 
                stop_threshold=1e-7, 
                ckpt_dir=os.path.join(os.path.abspath(''), 'models','Transformer'),
                 inference_mode=False, saved_model=None):
        if head_linear is None:
            head_linear = [dim_g, sequence_length*feature_channel_num]
        if tail_linear is None:
            tail_linear = [dim_s]
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if model_name is None:
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, model_name)
        self.model = self.create_model()
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')    # Set the BVL to large number
        self.best_training_loss = float('inf')    # Set the BTL to large number

        # marking the flag object with these information
        flags = object()
        flags.dim_g, flags.dim_s, flags.feature_channel_num, flags.nhead_encoder, flags.dim_fc_encoder, 
        flags.num_encoder_layer, flags.head_linear, flags.tail_linear, flags.sequence_length, 
        flags.model_name = dim_g, dim_s, feature_channel_num, nhead_encoder, dim_fc_encoder,num_encoder_layer, 
        head_linear, tail_linear, sequence_length, model_name
        self.flags = flags

    def create_model(self):
        """
        Function to create the network module
        :return: the created nn module
        """
        model = Transformer(self.flags)
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

    def make_optimizer(self, optim, lr, reg_scale):
        """
        Make the corresponding optimizer from the Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg_scale)
        elif optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=reg_scale)
        elif optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op
    
    def make_lr_scheduler(self, optm, lr_scheduler, lr_decay_rate, warm_restart_T_0=50):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        if lr_scheduler == 'warm_restart':
            return lr_scheduler.CosineAnnealingWarmRestarts(optm, warm_restart_T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False) 
        elif lr_scheduler == 'reduce_plateau':
            return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
    
    
    def load_pretrain_from_paper(self, dataset_name):
        """
        This is the function that loads the pre-trained weigths from the paper that the authors optimized
        """
        print("Sorry this function is still being updated, check back later for more details!")
        return 0


    def load_model(self, model_directory=None):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        if model_directory is None:
            model_directory = self.ckpt_dir
        # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
        self.model = torch.load(os.path.join(model_directory, 'best_model_forward.pt'))

    def train(self, train_loader, test_loader, epochs=500, optm='Adam', reg_scale=5e-4,
            lr=1e-3, lr_scheduler='reduce_plateau', lr_decay_rate=0.3, eval_step=10):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        print("Starting training now")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer(optm, lr, reg_scale)
        self.lr_scheduler = self.make_lr_scheduler(self.optm, lr_scheduler, lr_decay_rate)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))
        
        for epoch in range(epochs):
            # Set to Training Mode
            train_loss = 0
            self.model.train()
            for j, (geometry, spectra) in enumerate(train_loader):
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

            # Calculate the avg loss of training
            train_avg_loss = train_loss / (j + 1)
            # Recording the best training loss
            if train_avg_loss < self.best_training_loss:
                self.best_training_loss = train_avg_loss

            if epoch % eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/total_train', train_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()

                    S_pred = self.model(geometry)
                    loss = self.make_loss(logit=S_pred, labels=spectra)
                    test_loss += loss.cpu().data.numpy()                                       # Aggregate the loss
                    del loss, S_pred

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

                    if self.best_validation_loss < self.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        tk.record(1)                # Record the total time of the training peroid
    
    def __call__(self, test_X):
        """
        This is to call this model to do testing, 
        :param: test_X: The testing X to be input to the model
        """
        # put model to GPU if cuda available
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
            test_X = test_X.cuda()
        # Make the model to eval mode
        self.model.eval()
        # output the Ypred
        Ypred = self.model(test_X).cpu().data.numpy()
        return Ypred

    def evaluate(self, test_x, test_y,  save_dir='data/', prefix=''):
        # Make sure there is a place for the evaluation
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # Make the names align
        geometry, spectra = test_x, test_y

        # Put things on conda
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
            if cuda:
                geometry = geometry.cuda()
                spectra = spectra.cuda()
            Ypred = self.model(geometry).cpu().data.numpy()
            np.savetxt(fxt, geometry.cpu().data.numpy())
            np.savetxt(fyt, spectra.cpu().data.numpy())
            np.savetxt(fyp, Ypred)
        tk.record(1)                # Record the total time of the eval period

        MSE = plotMSELossDistrib(Ypred_file, Ytruth_file)
        return MSE
