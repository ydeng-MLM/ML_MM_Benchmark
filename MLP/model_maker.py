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
    def __init__(self, flags, fre_low=0.8, fre_high=1.5):
        super(Forward, self).__init__()

        self.skip_connection = flags.skip_connection
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

        if self.use_conv:
            # Conv Layer definitions here
            self.convs = nn.ModuleList([])
            self.bn_convs = nn.ModuleList([])
            in_channel = 1                                                  # Initialize the in_channel number
            for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                         flags.conv_kernel_size,
                                                                         flags.conv_stride)):
                if stride == 2:     # We want to double the number
                    pad = int(kernel_size/2 - 1)
                elif stride == 1:   # We want to keep the number unchanged
                    pad = int((kernel_size - 1)/2)
                else:
                    Exception("Now only support stride = 1 or 2, contact Ben")

                self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                    stride=stride, padding=pad)) # To make sure L_out double each time
                self.bn_convs.append(nn.BatchNorm1d(out_channel))
                in_channel = out_channel # Update the out_channel


            self.convs.append(nn.Conv1d(4, out_channels=1, kernel_size=1, stride=1, padding=0))        #Questionable if we still need the last conv layer, will do some test
            #self.bn_convs.append(nn.BatchNorm1d(1))

            #self.fc_out = nn.Linear(200, 3)

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out

        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size()
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
            else:
                if ind < len(self.linears) - 1:
                    out = F.relu(bn(fc(out)))
                else:
                    out = fc(out)
                

        # The normal mode to train without Lorentz
        if self.use_conv:
            out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
            # For the conv part
            for ind, conv in enumerate(self.convs):
                out = conv(out)

            out = out.squeeze(1)
            #print(out.shape)
            # Final touch, because the input is normalized to [-1,1]
            # S = tanh(out.squeeze())
            #out = out.squeeze()
        return out

