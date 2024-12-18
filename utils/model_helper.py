
""" helper function for nn.Module """
import torch
import torch.nn as nn


class ModelUtils(object):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel()
                            for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    @staticmethod
    def get_memory_usage(model, input_shape):
        from torchstat import stat
        result = stat(model, input_shape)
        print(result)

    @staticmethod
    def print_model_layer(model):
        for name, value in model.named_parameters():
            print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

    @staticmethod
    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint, model, optimizer):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])



def get_regularization(model, ord = 1, lambda_value = 0.1):
    reg_loss = 0.0
    # l2_lambda  = 0.1
    for param in model.parameters():
        reg_loss += torch.linalg.norm(param, ord = ord)
    return lambda_value * reg_loss


# def init_linear_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.weight.data = m.weight.data*weight_multiplyer


def get_lr(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append( param_group['lr'])
    return lr_list