#%%
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from models.FC import FCNet_H, FCNet_H_class

#%%
class ContrastiveEncoder(nn.Module):
    def __init__(self, feature_channel, 
                        hidden_list ,
                        h_dim = 128,
                        if_bn = False,
                        if_dp = False):
        super().__init__() 
        self.encoder = FCNet_H(feature_channel = feature_channel, 
                                output_channel = h_dim,
                                hidden_list = hidden_list,
                                if_bn = if_bn,
                                if_dp = if_dp)

    def forward(self, x):
        return self.encoder(x)


class ContrastiveDescriminator(nn.Module):
    def __init__(self, feature_channel,
                 hidden_list ,
                 h_dim = 128,
                 if_bn = False,
                 if_dp = False):
        super().__init__()
        self.encoder = FCNet_H(feature_channel = feature_channel,
                               output_channel = h_dim,
                               hidden_list = hidden_list,
                               if_bn = if_bn,
                               if_dp = if_dp)

    def forward(self, x):
        return self.encoder(x)


class ContrastiveDescriminatorClass(nn.Module):
    def __init__(self, feature_channel, 
                        hidden_list ,
                        h_dim = 128,
                        if_bn = False,
                        if_dp = False):
        super().__init__() 
        self.encoder = FCNet_H_class(feature_channel = feature_channel,
                                output_channel = h_dim,
                                hidden_list = hidden_list,
                                if_bn = if_bn,
                                if_dp = if_dp)

    def forward(self, x):
        return self.encoder(x)