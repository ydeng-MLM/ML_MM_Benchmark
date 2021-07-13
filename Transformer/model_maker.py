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

        # The slicing method
        assert  (flags.dim_G % flags.sequence_length == 0), 'Assertion error, your dim_G  should be divisible by the flags.sequence_length'
        self.sequence_fc_layer = nn.Linear(int(flags.dim_G / flags.sequence_length) , flags.feature_channel_num)
        
        # Transformer Encoder module
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=flags.feature_channel_num, nhead=flags.nhead_encoder,
                                                    dim_feedforward=flags.dim_fc_encoder)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                num_layers=flags.num_encoder_layer)

        # For the no decoder case, the output spectra
        # self.decoder = nn.Linear(flags.dim_G * flags.feature_channel_num, flags.dim_S)

        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])

        # Due to the two possible ways of preparing the sequences, the number of nodes at the last layer varies
        if self.flags.sequence_length == 1:
            last_layer_node_num = flags.dim_G * flags.feature_channel_num
        else:
            last_layer_node_num = flags.sequence_length * flags.feature_channel_num

        if flags.linear:
            self.linears.append(nn.Linear(last_layer_node_num, flags.linear[0]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[0]))
        else:
            self.linears.append(nn.Linear(last_layer_node_num, flags.dim_S))
            self.bn_linears.append(nn.BatchNorm1d(flags.dim_S))

        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))


    def forward(self, G):
        """
        The forward function of the transformer
        """
        
        if self.flags.sequence_length == 1:
            """
            In this case we are using the 1x1 convolution way to change the 
            [dim_G x 1] to [sentence_length (10) x embedding_length(512)], which is [dim_G x flags.feature_channel_num]
            """
            # print('original G shape', G.size())
            out = torch.unsqueeze(G, 1)         # Batch_size x 1 x dim_G
            # Get the num channels to be more than 1 channel
            out = self.conv1d_layer(out)
            # print('after conv1d ', out.size())
            out = out.permute(0, 2, 1)
        else:
            """
            In this case we are using the MLP mixer approach of changing the
            [dim_G x 1] to [sentence_length (10) x embedding_length(512)], which is [flags.sequence_length x flags.feature_channel_num]
            """
            out = torch.unsqueeze(G, 2)         # Batch_size x dim_G x 1
            # print('after expansion ', out.size())
            out = out.view([out.size(0), self.flags.sequence_length, -1])
            # print('after reshape ', out.size())
            out = self.sequence_fc_layer(out)
            # print('after sequence_fc ', out.size())
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