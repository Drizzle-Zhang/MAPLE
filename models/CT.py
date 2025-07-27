#%%
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from models.FC import FCNet_H, FCNet_H_class

#%%
class ContrastiveEncoder(nn.Module):
    """
    Encoder model for contrastive learning.
    
    This model transforms the input methylation data into a latent representation
    that can be used for downstream tasks like age prediction and disease risk assessment.
    
    Attributes:
        encoder: Neural network that encodes the input features into a latent space
    """
    def __init__(self, feature_channel, 
                        hidden_list ,
                        h_dim = 128,
                        if_bn = False,
                        if_dp = False):
        """
        Initialize the contrastive encoder.
        
        Args:
            feature_channel (int): Number of input features (CpG sites)
            hidden_list (list): List of hidden layer dimensions
            h_dim (int): Dimension of the output latent representation
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
        """
        super().__init__() 
        self.encoder = FCNet_H(feature_channel = feature_channel, 
                                output_channel = h_dim,
                                hidden_list = hidden_list,
                                if_bn = if_bn,
                                if_dp = if_dp)

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of methylation data
            
        Returns:
            torch.Tensor: Encoded latent representation
        """
        return self.encoder(x)


class ContrastiveDescriminator(nn.Module):
    """
    Discriminator model for contrastive learning (regression).
    
    This model is used in the contrastive learning framework to discriminate
    between different samples based on their latent representations.
    
    Attributes:
        encoder: Neural network that processes the latent representations
    """
    def __init__(self, feature_channel,
                 hidden_list ,
                 h_dim = 128,
                 if_bn = False,
                 if_dp = False):
        """
        Initialize the contrastive discriminator.
        
        Args:
            feature_channel (int): Number of input features
            hidden_list (list): List of hidden layer dimensions
            h_dim (int): Dimension of the output
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
        """
        super().__init__()
        self.encoder = FCNet_H(feature_channel = feature_channel,
                               output_channel = h_dim,
                               hidden_list = hidden_list,
                               if_bn = if_bn,
                               if_dp = if_dp)

    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Args:
            x (torch.Tensor): Input tensor of latent representations
            
        Returns:
            torch.Tensor: Discriminator output
        """
        return self.encoder(x)


class ContrastiveDescriminatorClass(nn.Module):
    """
    Discriminator model for contrastive learning (classification).
    
    This model is used for classification tasks in the contrastive learning framework,
    such as disease risk prediction.
    
    Attributes:
        encoder: Neural network that processes the latent representations for classification
    """
    def __init__(self, feature_channel, 
                        hidden_list ,
                        h_dim = 128,
                        if_bn = False,
                        if_dp = False):
        """
        Initialize the contrastive discriminator for classification.
        
        Args:
            feature_channel (int): Number of input features
            hidden_list (list): List of hidden layer dimensions
            h_dim (int): Number of output classes
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
        """
        super().__init__() 
        self.encoder = FCNet_H_class(feature_channel = feature_channel,
                                output_channel = h_dim,
                                hidden_list = hidden_list,
                                if_bn = if_bn,
                                if_dp = if_dp)

    def forward(self, x):
        """
        Forward pass through the classification discriminator.
        
        Args:
            x (torch.Tensor): Input tensor of latent representations
            
        Returns:
            torch.Tensor: Classification probability logits
        """
        return self.encoder(x)