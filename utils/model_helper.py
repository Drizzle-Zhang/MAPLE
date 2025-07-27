import torch
import torch.nn as nn


class ModelUtils:
    """
    Utility class for neural network model operations.
    
    This class provides static methods for model analysis, checkpoint management,
    and other common model operations.
    """
    
    @staticmethod
    def get_parameter_number(model):
        """
        Count the total and trainable parameters in a model.
        
        Args:
            model (nn.Module): PyTorch model
            
        Returns:
            dict: Dictionary with 'Total' and 'Trainable' parameter counts
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    @staticmethod
    def get_memory_usage(model, input_shape):
        """
        Calculate memory usage of a model.
        
        Args:
            model (nn.Module): PyTorch model
            input_shape (tuple): Shape of the input tensor
            
        Note:
            Requires torchstat package
        """
        from torchstat import stat
        result = stat(model, input_shape)
        print(result)

    @staticmethod
    def print_model_layer(model):
        """
        Print layer names and their gradient requirements.
        
        Args:
            model (nn.Module): PyTorch model
        """
        for name, value in model.named_parameters():
            print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

    @staticmethod
    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        """
        Save model checkpoint to file.
        
        Args:
            state (dict): State dictionary containing model state
            filename (str): Path to save the checkpoint
        """
        print("=> Saving checkpoint")
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint, model, optimizer):
        """
        Load model checkpoint from file.
        
        Args:
            checkpoint (dict): Loaded checkpoint state dictionary
            model (nn.Module): Model to load the state into
            optimizer: Optimizer to load the state into
        """
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])


def get_regularization(model, ord=1, lambda_value=0.1):
    """
    Calculate regularization loss for a model.
    
    Args:
        model (nn.Module): PyTorch model
        ord (int): Order of the norm (1 for L1, 2 for L2)
        lambda_value (float): Regularization strength
        
    Returns:
        torch.Tensor: Regularization loss
    """
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.linalg.norm(param, ord=ord)
    return lambda_value * reg_loss


# def init_linear_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.weight.data = m.weight.data*weight_multiplyer


def get_lr(optimizer):
    """
    Get current learning rates from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        list: List of learning rates for each parameter group
    """
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


class LossSoft(nn.Module):
    def __init__(self, lambda_1=1, epsilon=0.1):
        super(LossSoft, self).__init__()
        self.lambda_1 = lambda_1
        self.epsilon = epsilon

    def forward(self, pred_age, true_age):
        diff_age = pred_age - true_age
        # diff_age.requires_grad = True
        interval_1 = (diff_age < -self.epsilon)
        interval_2 = (diff_age > self.epsilon)
        loss_total = torch.mean(-torch.pow(diff_age, 1) * interval_1) + \
                     torch.mean(torch.pow(diff_age, 1) * interval_2) * self.lambda_1
        return loss_total
