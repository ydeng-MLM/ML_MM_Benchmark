import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from einops.layers.torch import Rearrange

from . import helper

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
    def __init__(self,input_dim,output_dim,mlp_dim,patch_size,mixer_layer_num, \
                embed_dim=128, token_dim=128, channel_dim=256, mlp_layer_num_front=3,mlp_layer_num_back=3,dropout=0.,device=None):
        super().__init__()

        # GPU device
        if not device:
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
          self.device = device

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

    def evaluate(self,x,y=None,criterion=None):
        self.eval()
        prediction = self.forward(x.to(self.device))
        if y is not None:
          if criterion is None:
            criterion = nn.MSELoss()

          error = criterion(prediction,y)
          return error
        return prediction

    def load_model(self,dataset):
      '''
      TO DO
      '''
      pass

    def train_(self,trainX,trainY,valX,valY, \
    batch_size=128,criterion=None,lr=1e-4,epochs=300, lr_scheduler=None, weight_decay=0.):
      '''
      Parameters: 
      (1) X: torch tensor
      (2) Y: torch tensor
      '''
      trainloader = torch.utils.data.DataLoader(helper.MyDataset(trainX,trainY), batch_size=batch_size)

      self.to(self.device)
      optimizer = optim.Adam(self.parameters(), lr=lr,weight_decay = weight_decay)
      if not criterion:
        criterion = nn.MSELoss()

      minvalloss = 1
      trainlosses=[]
      vallosses=[]
      for _ in tqdm(range(epochs)):

          self.train()
          for data in trainloader:
              x, y = data

              optimizer.zero_grad()
              predict = self.forward(x.to(self.device))
              loss = criterion(predict,y.to(self.device))
              loss.backward()

              optimizer.step()
          with torch.no_grad():
              trainlosses.append(loss.item())
              self.eval()
              valloss = criterion(self.forward(valX.to(self.device)),valY.to(self.device)).item()
              if valloss < minvalloss:
                  minvalloss = valloss
                  vallosses.append(valloss)

      return trainlosses,vallosses