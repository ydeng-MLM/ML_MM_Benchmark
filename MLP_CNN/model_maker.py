"""
This is the module where the model is defined. It uses the nn.Module as backbone to create the network structure
"""
# Own modules

# Built in
import math
# Libs
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt


class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()

        try:
            self.skip_connection = flags.skip_connection
        except:
            print("This is older version of code there is no skip flag")
            self.skip_connection = False
        self.use_conv = flags.use_conv
        if flags.dropout > 0:
            self.dp = True
            self.dropout = nn.Dropout(p=flags.dropout)
        else:
            self.dp = False
        self.skip_head = flags.skip_head

        """
        General layer definitions:
        """

        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        #self.dropout = nn.ModuleList([])       #Dropout layer was tested for fixing overfitting problem
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))
            #self.dropout.append(nn.Dropout(p=0.05))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out

        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            # If there are skip connections 
            if self.skip_connection:
                if ind < len(self.linears) - 1:
                    if ind == self.skip_head:
                        out = F.relu(bn(fc(out)))
                        if self.dp:
                            out = self.dropout(out)
                        identity = out
                    elif ind > self.skip_head and (ind - self.skip_head)%2 == 0:
                        out = F.relu(bn(fc(out)))   # ReLU + BN + Linear
                        if self.dp: 
                            out = self.dropout(out)
                        out += identity
                        identity = out
                    else:
                        out = F.relu(bn(fc(out)))
                        if self.dp:
                            out = self.dropout(out)
                else:
                    out = (fc(out))
            else:       # Cases where there are no skip connections
                if ind < len(self.linears) - 1:
                    out = F.relu(bn(fc(out)))
                else:
                    out = fc(out)

        return out

