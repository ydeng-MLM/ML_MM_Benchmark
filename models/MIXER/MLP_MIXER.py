import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from einops.layers.torch import Rearrange

from . import helper

import os
import math

class FeedForward(nn.Module):
  '''
  FC -> GELU + DO -> FC 
  '''
  def __init__(self, dim, hidden_dim, dropout = 0.):
      super().__init__()
      self.net = nn.Sequential(
          nn.Linear(dim, hidden_dim),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(hidden_dim, dim)
      )
  def forward(self, x):
      return self.net(x)

class MLayer(nn.Module):
  def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
    super().__init__()

    self.token_mix = nn.Sequential(
        nn.LayerNorm(dim),
        Rearrange('b n d -> b d n'),
        FeedForward(num_patch, token_dim, dropout),
        Rearrange('b d n -> b n d')
    )

    self.channel_mix = nn.Sequential(
        nn.LayerNorm(dim),
        FeedForward(dim, channel_dim, dropout),
    )

  def forward(self, x):
      #skip connection
      x = x + self.token_mix(x)
      x = x + self.channel_mix(x)

      return x  


class MMixer(nn.Module):
  def __init__(self,patch_size,embed_dim, n_block, token_dim, channel_dim,input_dim, output_dim,dropout=0.,expand=False,expand_dim=128):
    super().__init__()

    if expand:
      self.expand_linear = nn.Linear(input_dim,expand_dim)
      self.expand = True
    else:
      self.expand = False

    self.linear_embedding = nn.Linear(patch_size,embed_dim)
    
    if expand:
      assert expand_dim%patch_size == 0, "Impossible to patchify the input with this patch_size"
      self.num_patch = expand_dim//patch_size
    else:
      assert input_dim%patch_size == 0, "Impossible to patchify the input with this patch_size"
      self.num_patch = input_dim//patch_size
    
    self.patch_size = patch_size

    self.mixer_layers = nn.ModuleList([])
    for _ in range(n_block):
        self.mixer_layers.append(MLayer(embed_dim, self.num_patch, token_dim, channel_dim,dropout=dropout))

    self.layer_norm = nn.LayerNorm(embed_dim)

    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, output_dim)
    )


  def patchify(self,X,patch_size=1):
    '''
    suppose X is of shape (n,d): n is size of batch, d is size of feature
    '''
    n,d = X.shape
    assert d%patch_size == 0, "Impossible to patchify the input with this patch_size"
    num_patch = d//patch_size

    stack=[]  
    for i in range(num_patch):
      stack.append(X[:,i*patch_size:(i+1)*patch_size])
    return torch.stack(stack,dim=1) #the returned X is of size (n,num_patch,patch_size)
  
  def forward(self,x):
    #expand the input dimension if necessary
    if self.expand:
      x = self.expand_linear(x)

    #slice input into patches 
    x = self.patchify(x,self.patch_size)
    #linear embedding from patch_size to dim
    x = self.linear_embedding(x)

    for mixer in self.mixer_layers:
      x = mixer(x)

    x = self.layer_norm(x)
    x = x.mean(dim=1) #global average pooling
    prediction = self.mlp(x)

    return prediction

class Monster(nn.Module):
    '''
        OOP for the model that combines and Mixer and MLP layers
    '''
    def __init__(self,input_dim,output_dim,mlp_dim,patch_size,mixer_layer_num,mlp1_layer_num=3,dropout=0.):
        super().__init__()
        sequence=[nn.Linear(input_dim,mlp_dim),nn.ReLU(),nn.Dropout(dropout)]
        for _ in range(mlp1_layer_num):
            sequence.append(nn.Linear(mlp_dim,mlp_dim))
            sequence.append(nn.ReLU())
            sequence.append(nn.Dropout(dropout))
        
        self.MLP1 = nn.Sequential(
            *sequence
        )
        
        #the mixer takes output from first MLP 
        self.mixer = MMixer(patch_size=patch_size,embed_dim=128,n_block=mixer_layer_num,
        token_dim=128, channel_dim=256, input_dim=mlp_dim,output_dim=output_dim,expand=False,dropout=dropout)

    def forward(self,x):
        x = self.MLP1(x)
        prediction = self.mixer(x)

        return prediction

#=========== Define Model unique to FB search ============
class MonsterFB(nn.Module):
    '''
        OOP for the model that combines and Mixer and MLP layers
    '''
    def __init__(self,dim_g,dim_s,mlp_dim,patch_size,mixer_layer_num, \
                embed_dim=128, token_dim=128, channel_dim=256, \
                mlp_layer_num_front=3,mlp_layer_num_back=3,dropout=0., \
                device=None, stop_threshold=1e-7, \
                log_mode=False, ckpt_dir= os.path.join(os.path.abspath(''), 'models','Mixer') \
                ):
        super().__init__()

        input_dim = dim_g
        output_dim = dim_s
        # GPU device
        if not device:
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
          self.device = device

        self.stop_threshold = stop_threshold
        self.ckpt_dir = ckpt_dir
        self.log = SummaryWriter(self.ckpt_dir)
        self.log_mode = log_mode

        # MLP layers in front
        sequence1=[nn.Linear(input_dim,mlp_dim),nn.ReLU(),nn.Dropout(dropout)]
        for _ in range(mlp_layer_num_front-1):
            sequence1.append(nn.Linear(mlp_dim,mlp_dim))
            sequence1.append(nn.ReLU())
            sequence1.append(nn.Dropout(dropout))

        # MLP layers at back
        if mlp_layer_num_back == 1:
            sequence2=[nn.Linear(mlp_dim,output_dim)]
        else:
            sequence2=[nn.Linear(mlp_dim,mlp_dim),nn.ReLU(),nn.Dropout(dropout)]
            for _ in range(mlp_layer_num_back-1):
                if _ == mlp_layer_num_back -2:
                    sequence2.append(nn.Linear(mlp_dim,output_dim))
                else:
                    sequence2.append(nn.Linear(mlp_dim,mlp_dim))
                    sequence2.append(nn.ReLU())
                    sequence2.append(nn.Dropout(dropout))
        
        self.MLP1 = nn.Sequential(
            *sequence1
        )
        self.MLP2 = nn.Sequential(
            *sequence2
        )

        
        #the MLP-MIXER
        self.mixer = MMixer(patch_size=patch_size,embed_dim=embed_dim,n_block=mixer_layer_num,
        token_dim=token_dim, channel_dim=channel_dim, input_dim=mlp_dim,output_dim=mlp_dim,expand=False)

    def forward(self,x):
        x = self.MLP1(x)
        x = self.mixer(x)
        prediction = self.MLP2(x)

        return prediction

    def evaluate(self, test_x, test_y,  save_dir='data/', prefix=''):
        # Make sure there is a place for the evaluation
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        saved_model_str = prefix
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))

        test_loader = DataLoader(helper.MyDataset(test_x,test_y))
        if criterion is None:
            criterion = nn.MSELoss()
        mse_error = helper.eval_loader(self,test_loader,self.device,criterion)
        
        # Write to files if log_mode = True
        if self.log_mode:
          with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                  open(Ypred_file, 'a') as fyp:
              for j, (geometry, spectra) in enumerate(test_loader):
                  
                  geometry = geometry.to(self.device)
                  spectra = spectra.to(self.device)
                  Ypred = self.forward(geometry).cpu().data.numpy()
                  np.savetxt(fxt, geometry.cpu().data.numpy())
                  np.savetxt(fyt, spectra.cpu().data.numpy())
                  np.savetxt(fyp, Ypred)
        
        return mse_error

    def load_model(self, pre_trained_model=None, model_directory=None):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        if pre_trained_model is None:       # Loading the trained model
            if model_directory is None:
                model_directory = self.ckpt_dir
            # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
            self.model = torch.load(os.path.join(model_directory, 'best_model_forward.pt'))
            print("You have successfully loaded the model from ", model_directory)
        else:       # Loading the pretrained model from the internet
            print("You have successfully loaded the pretrained model for ", pre_trained_model)

    def train_(self,trainloader,testloader, \
    batch_size=128,criterion=None,epochs=300,  eval_step=10, \
    optm='Adam', lr=1e-4, weight_decay=5e-4, lr_scheduler_name=None, lr_decay_rate=0.3):
        '''
        Parameters: 
        (1) trainloader: data loader of training data
        (2) testloader: data loader of test/val data
        '''

        # Construct optimizer after the model moved to GPU
        optimizer = self.make_optimizer(optim=optm, lr=lr, reg_scale=weight_decay)
        scheduler = self.make_lr_scheduler(optimizer, lr_scheduler_name, lr_decay_rate)
        
        if not criterion:
          criterion = nn.MSELoss()
        
        minvalloss = math.inf
        self.to(self.device)
        for epoch in tqdm(range(epochs)):

            self.train()
            for i,data in enumerate(trainloader):
                x, y = data

                optimizer.zero_grad()
                predict = self.forward(x.to(self.device))
                loss = criterion(predict,y.to(self.device))
                loss.backward()

                optimizer.step()
            
            if epoch % eval_step == 0:                      # For eval steps, do the evaluations and tensor board
                
                trainloss = helper.eval_loader(self,trainloader,self.device,criterion)
                valloss = helper.eval_loader(self,testloader,self.device,criterion)
               
                self.log.add_scalar('Loss/total_train', trainloss, epoch)
                self.log.add_scalar('Loss/total_test', valloss, epoch)
                
                if valloss < minvalloss:
                    minvalloss = valloss
                    self.minvalloss = minvalloss
                    if self.log_mode:
                      self.save()
                      print("Saving the model down...")

                    if minvalloss < self.stop_threshold:
                      print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.minvalloss))
                      break

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, trainloss, valloss))
            
            if scheduler:
              scheduler.step()

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
    
    def make_lr_scheduler(self, optm, lr_scheduler_name, lr_decay_rate, warm_restart_T_0=50):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        if lr_scheduler_name == 'warm_restart':
            return lr_scheduler.CosineAnnealingWarmRestarts(optm, warm_restart_T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False) 
        elif lr_scheduler_name == 'reduce_plateau':
            return lr_scheduler_name.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)
        else:
            return None

    def make_optimizer(self, optim, lr, reg_scale):
        """
        Make the corresponding optimizer from the Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if optim == 'Adam':
            op = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg_scale)
        elif optim == 'RMSprop':
            op = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=reg_scale)
        elif optim == 'SGD':
            op = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op