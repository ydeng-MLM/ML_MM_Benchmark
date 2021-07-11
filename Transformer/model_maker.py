# This is the model maker for the Transformer model

# From Built-in
from time import time

# From libs
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class Transformer(nn.Module):
    """
    The constructor for Transformer network
    :param flags: inut flags from configuration
    :return: The transformer  network
    """
    def __init__(self, flags):
        super(Transformer, self).__init__()
        # Save the flags
        self.flags = flags

        # The 1x1 convolution module to make more channel from Geometry
        self.conv1d_layer = nn.Conv1d(in_channels=1, out_channels=flags.feature_channel_num, kernel_size=1)

        # Transformer Encoder module
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=flags.feature_channel_num, nhead=flags.nhead_encoder,
                                                    dim_feedforward=flags.dim_fc_encoder)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                num_layers=flags.num_encoder_layer)

        # For the no decoder case, the output spectra
        # self.decoder = nn.Linear(flags.dim_G * flags.feature_channel_num, flags.dim_S)

        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])

        if flags.linear:
            self.linears.append(nn.Linear(flags.dim_G * flags.feature_channel_num, flags.linear[0]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[0]))
        else:
            self.linears.append(nn.Linear(flags.dim_G * flags.feature_channel_num, flags.dim_S))
            self.bn_linears.append(nn.BatchNorm1d(flags.dim_S))

        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))


    def forward(self, G):
        """
        The forward function of the transformer
        """
        # print('original G shape', G.size())
        out = torch.unsqueeze(G, 1)
        # print('after squeezing ', out.size())
        # Get the num channels to be more than 1 channel
        out = self.conv1d_layer(out)
        # print('after conv1d ', out.size())
        out = out.permute(0, 2, 1)
        # print('after permuting', out.size())
        # Get the the data transformed
        out = self.transformer_encoder(out)
        # print('after transformer ', out.size())
        out = out.view([out.size(0), -1])
        # print('after viewing ', out.size())
        # Get the transformed feature to the decoder and get spectra
        # out = self.decoder(out)
        # print('after decoder ', out.size())
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = fc(out) 
        return out