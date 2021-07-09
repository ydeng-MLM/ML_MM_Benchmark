# This is the model maker for the Transformer model

# From Built-in
from time import time

# From libs
import torch
import torch.nn as nn
import torch.optim
import numpy as np
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
        self.decoder = nn.Linear(flags.dim_G * flags.feature_channel_num, flags.dim_S)


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
        # Get the the data transformed
        out = self.transformer_encoder(out)
        # print('after transformer ', out.size())
        out = out.view([out.size(0), -1])
        # print('after viewing ', out.size())
        # Get the transformed feature to the decoder and get spectra
        out = self.decoder(out)
        # print('after decoder ', out.size())

        return out
