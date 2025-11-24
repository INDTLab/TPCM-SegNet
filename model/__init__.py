import torch.utils.model_zoo as model_zoo
import torch

from model.project.TPCMSegNet import TPCMSegNet


def model_bulider(model, in_channels, num_class):

    if model == 'TPCMSegNet':
        model = TPCMSegNet (
        input_channels = in_channels,
        num_classes = num_class,
        depths=[2, 4, 4, 8],
        dim=64,
        in_dim=64
    )
         
    else: 
        raise ValueError(f'Unknown model {model}')
    
    return model

