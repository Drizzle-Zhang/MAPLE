import torch
import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    """
    Basic building block for fully connected networks.
    
    Each block consists of a linear layer, optional batch normalization,
    ReLU activation, and optional dropout for regularization.
    
    Attributes:
        fc1: Linear transformation layer
        bn1: Batch normalization layer (optional)
        relu: ReLU activation function
        dropout: Dropout layer (optional)
    """
    def __init__(self, input_size, output_size, 
                if_bn=False, if_dp=False):
        """
        Initialize a basic building block.
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
        """
        super(block, self).__init__()
        self.if_bn = if_bn
        self.if_dp = if_dp
        self.fc1 = nn.Linear(input_size, output_size)

        if self.if_bn:
            self.bn1 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

        if self.if_dp:
            self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Forward pass through the block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after linear transformation, optional batch norm,
                         ReLU activation, and optional dropout
        """
        x = self.fc1(x)
        if self.if_bn:
            x = self.bn1(x)
        x = self.relu(x)

        if self.if_bn:
            x = self.bn1(x)
        if self.if_dp:
            x = self.dropout(x)
        return x


class FCNet(nn.Module):
    """
    Fully connected neural network with configurable hidden layers.
    
    This network has a fixed number of hidden layers with the same dimension.
    
    Attributes:
        first_layer: First block in the network
        mediate_layer: Middle blocks in the network
        last_layer: Final linear layer for output
    """
    def __init__(self, feature_channel, output_channel,
                 hidden_number, hidden_size,
                 if_bn=True, if_dp=True):
        """
        Initialize a fully connected network.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output features
            hidden_number (int): Number of hidden layers
            hidden_size (int): Size of each hidden layer
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
        """
        super(FCNet, self).__init__()
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.if_bn = if_bn

        self.first_layer = block(feature_channel, hidden_size, 
                                if_bn=if_bn, if_dp=if_dp)

        layers = []
        for i in range(hidden_number):
            layers.append(block(hidden_size, hidden_size, 
                              if_bn=if_bn, if_dp=if_dp))
        self.mediate_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(
            hidden_size, output_channel)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.first_layer(x)
        x = self.mediate_layer(x)
        x = self.last_layer(x)
        return x


class FCNet_H(nn.Module):
    """
    Fully connected neural network with configurable hidden layer sizes.
    
    This network allows different sizes for each hidden layer.
    
    Attributes:
        first_layer: First block in the network
        mediate_layer: Middle blocks in the network
        last_layer: Final linear layer for output
        tissue_embedding_model: Embedding layer for tissue types
        sex_embedding_model: Embedding layer for sex
    """
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp):
        """
        Initialize a fully connected network with configurable hidden layers.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output features
            hidden_list (list): List of hidden layer sizes
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
        """
        super(FCNet_H, self).__init__()
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.hidden_list = hidden_list

        self.first_layer = block(feature_channel, hidden_list[0], if_bn, if_dp)

        layers = []
        for i in range(len(hidden_list) - 1):
            layers.append(block(hidden_list[i], hidden_list[i+1], if_bn, if_dp))
        self.mediate_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(
            hidden_list[-1], output_channel)

        # Embedding layers for metadata (not used in the forward pass)
        self.tissue_embedding_model = nn.Embedding(num_embeddings=32, 
                                                  embedding_dim=32)
                                 
        self.sex_embedding_model = nn.Embedding(num_embeddings=4, 
                                               embedding_dim=32)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.first_layer(x)
        x = self.mediate_layer(x)
        x = self.last_layer(x)
        return x


class FCNet_H_class(nn.Module):
    """
    Fully connected neural network for classification tasks.
    
    Similar to FCNet_H but applies softmax to the output for classification.
    
    Attributes:
        first_layer: First block in the network
        mediate_layer: Middle blocks in the network
        last_layer: Final linear layer for output
        tissue_embedding_model: Embedding layer for tissue types
        sex_embedding_model: Embedding layer for sex
    """
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp):
        """
        Initialize a fully connected network for classification.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output classes
            hidden_list (list): List of hidden layer sizes
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
        """
        super(FCNet_H_class, self).__init__()
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.hidden_list = hidden_list

        self.first_layer = block(feature_channel, hidden_list[0], if_bn, if_dp)

        layers = []
        for i in range(len(hidden_list) - 1):
            layers.append(block(hidden_list[i], hidden_list[i+1], if_bn, if_dp))
        self.mediate_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(
            hidden_list[-1], output_channel)

        # Embedding layers for metadata (not used in the forward pass)
        self.tissue_embedding_model = nn.Embedding(num_embeddings=32,
                                                  embedding_dim=32)

        self.sex_embedding_model = nn.Embedding(num_embeddings=4,
                                               embedding_dim=32)

    def forward(self, x):
        """
        Forward pass through the network with softmax output.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Classification probability logits
        """
        x = self.first_layer(x)
        x = self.mediate_layer(x)
        x = self.last_layer(x)
        return F.log_softmax(x, dim=1)




class FCNet_Embedding(nn.Module):
    """
    Fully connected network with embedding layers for metadata.
    
    This network can incorporate tissue and sex embeddings into the feature vector.
    
    Attributes:
        if_embedding: Whether to use embeddings
        if_norm: Whether to normalize the output
        tissue_embedding_model: Embedding layer for tissue types
        sex_embedding_model: Embedding layer for sex
        fc_model: Core fully connected network
    """
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding=False, if_norm=False):
        """
        Initialize a fully connected network with embedding layers.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output features
            hidden_list (list): List of hidden layer sizes
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
            if_embedding (bool): Whether to use tissue and sex embeddings
            if_norm (bool): Whether to normalize the output
        """
        super(FCNet_Embedding, self).__init__()
        
        self.if_embedding = if_embedding
        self.if_norm = if_norm

        if self.if_embedding:
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings=32, 
                                    embedding_dim=embedding_dim)
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings=4, 
                                    embedding_dim=embedding_dim)

            self.fc_model = FCNet_H(feature_channel + embedding_dim*2, output_channel,
                 hidden_list, if_bn, if_dp) 
        else:
            self.fc_model = FCNet_H(feature_channel, output_channel,
                 hidden_list, if_bn, if_dp)

    def forward(self, feature, additional):
        """
        Forward pass through the network.
        
        Args:
            feature (torch.Tensor): Input feature tensor
            additional (dict): Dictionary containing metadata ('tissue_index' and 'sex_index')
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.if_embedding:
            device = feature.device
            tissue_index = additional["tissue_index"].to(device)
            sex_index = additional["sex_index"].to(device)
            tissue_embedding = self.tissue_embedding_model(tissue_index)
            sex_embedding = self.sex_embedding_model(sex_index)
            # tissue_embedding = self.tissue_embedding_model(tissue_index) * 7.0
            # sex_embedding = self.sex_embedding_model(sex_index)* 7.0
            feature_concat = torch.concat([feature, tissue_embedding, sex_embedding], axis=1)
            result = self.fc_model(feature_concat)
        else:
            result = self.fc_model(feature)
        
        # Normalize output if required
        if self.if_norm:
            result = result / torch.sqrt(torch.norm(result, p=1, dim=0))
        return result





class FCNet_Embedding_Mask(nn.Module):
    """
    Fully connected network with embedding layers and feature masking.
    
    This network selects a subset of input features using a learnable mask.
    
    Attributes:
        if_embedding: Whether to use embeddings
        if_norm: Whether to normalize the output
        mask: Learnable mask for feature selection
        sigmoid: Sigmoid function for mask values
        limit_feature_number: Maximum number of features to select
        tissue_embedding_model: Embedding layer for tissue types
        sex_embedding_model: Embedding layer for sex
        fc_model: Core fully connected network
    """
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding=False, if_norm=False, limit_feature_number=200):
        """
        Initialize a fully connected network with feature masking.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output features
            hidden_list (list): List of hidden layer sizes
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
            if_embedding (bool): Whether to use tissue and sex embeddings
            if_norm (bool): Whether to normalize the output
            limit_feature_number (int): Maximum number of features to select
        """
        super(FCNet_Embedding_Mask, self).__init__()
        
        self.if_embedding = if_embedding
        self.if_norm = if_norm

        # Initialize learnable mask
        self.mask = torch.randn(feature_channel)
        self.mask.requires_grad = True
        self.sigmoid = torch.nn.Sigmoid()
        self.limit_feature_number = limit_feature_number

        if self.if_embedding:
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings=32, 
                                    embedding_dim=embedding_dim)
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings=4, 
                                    embedding_dim=embedding_dim)

            self.fc_model = FCNet_H(feature_channel + embedding_dim*2, output_channel,
                 hidden_list, if_bn, if_dp) 
        else:
            self.fc_model = FCNet_H(feature_channel, output_channel,
                 hidden_list, if_bn, if_dp)

    def forward(self, feature, additional):
        """
        Forward pass through the network with feature masking.
        
        Args:
            feature (torch.Tensor): Input feature tensor
            additional (dict): Dictionary containing metadata ('tissue_index' and 'sex_index')
            
        Returns:
            tuple: (result, mask_vector) containing the output tensor and applied mask
        """
        # Apply mask to select top features
        mask_vector = self.sigmoid(self.mask) 
        vals, idx = mask_vector.topk(self.limit_feature_number)

        mask_vector = torch.zeros_like(feature)
        mask_vector[:,idx] = vals

        masked_feature = feature * mask_vector

        if self.if_embedding:
            device = feature.device
            tissue_index = additional["tissue_index"].to(device)
            sex_index = additional["sex_index"].to(device)

            tissue_embedding = self.tissue_embedding_model(tissue_index) 
            sex_embedding = self.sex_embedding_model(sex_index)

            # mask_vector = self.sigmoid(self.mask ) * self.limit_feature_number
            # masked_feature = feature * mask_vector
            feature_concat = torch.concat([masked_feature, tissue_embedding, sex_embedding], axis=1)
            result = self.fc_model(feature_concat)
        else:
            result = self.fc_model(masked_feature)
        
        # Normalize output if required
        if self.if_norm:
            result = result / torch.sqrt(torch.norm(result, p=1, dim=0))
        return result, mask_vector




class FCNet_Embedding_Mask2(nn.Module):
    """
    Fully connected network with embeddings and dimensionality reduction.
    
    This network applies a linear projection to reduce feature dimensions before processing.
    
    Attributes:
        if_embedding: Whether to use embeddings
        if_norm: Whether to normalize the output
        linear1: Linear layer for dimensionality reduction
        linear_down: Reduced dimension size
        tissue_embedding_model: Embedding layer for tissue types
        sex_embedding_model: Embedding layer for sex
        fc_model: Core fully connected network
    """
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding=False, if_norm=False):
        """
        Initialize a fully connected network with dimensionality reduction.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output features
            hidden_list (list): List of hidden layer sizes
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
            if_embedding (bool): Whether to use tissue and sex embeddings
            if_norm (bool): Whether to normalize the output
        """
        super(FCNet_Embedding_Mask2, self).__init__()
        
        self.if_embedding = if_embedding
        self.if_norm = if_norm
        self.linear_down = 1024
        self.linear1 = nn.Linear(feature_channel, self.linear_down)

        if self.if_embedding:
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings=32, 
                                    embedding_dim=embedding_dim)
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings=4, 
                                    embedding_dim=embedding_dim)

            self.fc_model = FCNet_H(self.linear_down + embedding_dim*2, output_channel,
                 hidden_list, if_bn, if_dp) 
        else:
            self.fc_model = FCNet_H(self.linear_down, output_channel,
                 hidden_list, if_bn, if_dp)

    def forward(self, feature, additional):
        """
        Forward pass through the network with dimensionality reduction.
        
        Args:
            feature (torch.Tensor): Input feature tensor
            additional (dict): Dictionary containing metadata ('tissue_index' and 'sex_index')
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Reduce feature dimensions
        feature_dimdown = self.linear1(feature)
        
        if self.if_embedding:
            device = feature.device
            tissue_index = additional["tissue_index"].to(device)
            sex_index = additional["sex_index"].to(device)
            tissue_embedding = self.tissue_embedding_model(tissue_index)
            sex_embedding = self.sex_embedding_model(sex_index)
            # tissue_embedding = self.tissue_embedding_model(tissue_index) * 7.0
            # sex_embedding = self.sex_embedding_model(sex_index)* 7.0
            feature_concat = torch.concat([feature_dimdown, tissue_embedding, sex_embedding], axis=1)
            result = self.fc_model(feature_concat)
        else:
            result = self.fc_model(feature_dimdown)
        
        # Normalize output if required
        if self.if_norm:
            result = result / torch.sqrt(torch.norm(result, p=1, dim=0))
        return result



class FCNet_2Embedding(nn.Module):
    """
    Fully connected network with dual embeddings.
    
    This network can incorporate tissue and sex embeddings from two different samples.
    
    Attributes:
        if_embedding: Whether to use embeddings
        tissue_embedding_model: Embedding layer for tissue types
        sex_embedding_model: Embedding layer for sex
        fc_model: Core fully connected network
    """
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding=False):
        """
        Initialize a fully connected network with dual embeddings.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output features
            hidden_list (list): List of hidden layer sizes
            if_bn (bool): Whether to use batch normalization
            if_dp (bool): Whether to use dropout for regularization
            if_embedding (bool): Whether to use tissue and sex embeddings
        """
        super(FCNet_2Embedding, self).__init__()
        
        self.if_embedding = if_embedding

        if self.if_embedding:
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings=32, 
                                    embedding_dim=embedding_dim)
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings=4, 
                                    embedding_dim=embedding_dim)

            self.fc_model = FCNet_H(feature_channel + embedding_dim * 4, output_channel,
                 hidden_list, if_bn, if_dp) 
        else:
            self.fc_model = FCNet_H(feature_channel, output_channel,
                 hidden_list, if_bn, if_dp)

    def forward(self, feature, additional1, additional2):
        """
        Forward pass through the network with dual embeddings.
        
        Args:
            feature (torch.Tensor): Input feature tensor
            additional1 (dict): Dictionary containing metadata for first sample
            additional2 (dict): Dictionary containing metadata for second sample
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.if_embedding:
            device = feature.device

            # Embeddings for first sample
            tissue_index1 = additional1["tissue_index"].to(device)
            sex_index1 = additional1["sex_index"].to(device)
            tissue_embedding1 = self.tissue_embedding_model(tissue_index1) * 7.0
            sex_embedding1 = self.sex_embedding_model(sex_index1) * 7.0

            # Embeddings for second sample
            tissue_index2 = additional2["tissue_index"].to(device)
            sex_index2 = additional2["sex_index"].to(device)
            tissue_embedding2 = self.tissue_embedding_model(tissue_index2) * 7.0
            sex_embedding2 = self.sex_embedding_model(sex_index2) * 7.0

            # Concatenate features and embeddings
            feature_concat = torch.concat([feature, tissue_embedding1, sex_embedding1,
                tissue_embedding2, sex_embedding2], axis=1)
        
            result = self.fc_model(feature_concat)
        else:
            result = self.fc_model(feature)
        return result




class FCNet_1(nn.Module):
    """
    Simple single-layer fully connected network.
    
    This network applies a single linear transformation to the input.
    
    Attributes:
        fc1: Linear layer
    """
    def __init__(self, feature_channel, output_channel):
        """
        Initialize a single-layer fully connected network.
        
        Args:
            feature_channel (int): Number of input features
            output_channel (int): Number of output features
        """
        super(FCNet_1, self).__init__()
        self.fc1 = nn.Linear(feature_channel, output_channel)

    def forward(self, x):
        """
        Forward pass through the single-layer network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.fc1(x)
        return x



if __name__ == "__main__":
    """
    Test code for the fully connected network models.
    This section demonstrates the usage of various network architectures.
    """
    # Test FCNet
    # models = FCNet(feature_channel=20, output_channel=2,
    #         hidden_number=1, hidden_size=10)
    # x = torch.rand(128, 20)
    # y = models(x)
    # print(y.shape)

    # Test FCNet_H
    # model2 = FCNet_H(feature_channel=20, output_channel=2, 
    #         hidden_list=[10,20,30,40], 
    #         if_bn=False, if_dp=False)
    # y = model2(x)
    # print(y.shape)

    # Test FCNet_1
    # model3 = FCNet_1(feature_channel=20, output_channel=2)
    # y = model2(x)
    # print(y.shape)

    # Test FCNet_Embedding
    model = FCNet_Embedding(feature_channel=20, output_channel=1,
                 hidden_list=[3,3,3], if_bn=False, if_dp=False)
    x = torch.rand(128, 20)
    y = model(x,x)
    print(y.shape)

    # Test feature mask and parameter optimization
    import torch.optim as optim

    model = FCNet_Embedding_Mask(feature_channel=500, output_channel=1,
                 hidden_list=[3,3,3], if_bn=False, if_dp=False, limit_feature_number=100)
    x = torch.rand(128, 500)
    y, mask_vector = model(x, x)
    print(y.shape)

    optimizer = optim.Adam([{"params": model.parameters()},
                            {"params": model.mask}], 
                            lr=0.01)
    print(mask_vector.shape)
    print(mask_vector)
    print(torch.sum(mask_vector))

    # Optimize the mask parameters
    for i in range(5000):
        y, mask_vector = model(x, x)
        loss = -torch.mean(torch.sum(mask_vector**2, dim=1))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(mask_vector[0,:])

#%%

# %%
