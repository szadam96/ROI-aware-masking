import albumentations as A
import cv2
import torch
import torch.nn as nn

from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from torchmetrics import R2Score
from transformers import VideoMAEConfig, VideoMAEForPreTraining
from models.architectures import CustomVideoMAEForPretraining
from models.architectures_new import CustomVideoMAEForRegression

from pytorchvideo.models.head import ResNetBasicHead
from pytorchvideo.models.byol import BYOL
from pytorchvideo.models.simclr import SimCLR
from pytorchvideo.models import r2plus1d, x3d
from torchvision import models

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import numpy as np

def get_augmentations(augmentation_params):
    '''Get the augmentation transforms
    
    Parameters
    ----------
    augmentation_params : dict
        Dictionary containing the augmentation parameters
    
    Returns
    -------
    data_transforms : dict
        Dictionary containing the augmentation transforms for training and validation
    '''
    augment_apply = augmentation_params['apply']
    img_size = augmentation_params['image_size']

    train_transforms = []
    validation_transforms = []

    if 'keep_ratio' in augment_apply:
        train_transforms.append(A.LongestMaxSize(max_size=img_size))
        train_transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0))
        validation_transforms.append(A.LongestMaxSize(max_size=img_size))
        validation_transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0))
    else:
        train_transforms.append(A.Resize(img_size, img_size))
        validation_transforms.append(A.Resize(img_size, img_size))

    if 'rotation' in augment_apply:
        train_transforms.append(A.SafeRotate(limit=augmentation_params.get('rotation_limit', 10), border_mode=cv2.BORDER_CONSTANT, value=0, p=1))

    if 'horizontal_flip' in augment_apply:
        train_transforms.append(A.HorizontalFlip(p=augmentation_params.get('horizontal_filp_p', 0.5)))

    if 'random_crop' in augment_apply:
        train_transforms.append(A.RandomResizedCrop(img_size, img_size, scale=(augmentation_params.get('random_crop_scale', 0.8),1)))

    if 'normalize' in augment_apply:
        #TODO parameterize
        #previous values: mean=0.1008, std=0.1826
        train_transforms.append(A.Normalize(0.0998, 0.1759))
        validation_transforms.append(A.Normalize(0.0998, 0.1759))

    if 'blur' in augment_apply:
        train_transforms.append(A.AdvancedBlur(p=augmentation_params.get('blur_p', 0.5), blur_limit=(3,7)))
        #validation_transforms.append(A.augmentations.transforms.AdvancedBlur(p=augmentation_params.get('blur_p', 1)))
    
    if 'brightness' in augment_apply:
        train_transforms.append(A.RandomBrightnessContrast(p=augmentation_params.get('brightness_p', 0.5),brightness_limit=augmentation_params.get('brightness_limit', 0.2), contrast_limit=augmentation_params.get('contrast_limit', 0.2)))
        #validation_transforms.append(A.augmentations.transforms.RandomBrightnessContrast(p=augmentation_params.get('brightness_p', 1)))
    
    if 'sharpen' in augment_apply:
        train_transforms.append(A.Sharpen(p=augmentation_params.get('sharpen_p', 0.5)))
        #validation_transforms.append(A.augmentations.transforms.Sharpen(p=augmentation_params.get('sharpen_p', 0.5)))
    
    if 'gamma' in augment_apply:
        train_transforms.append(A.RandomGamma(p=augmentation_params.get('gamma_p', 0.5), gamma_limit=augmentation_params.get('gamma_limit', (80,120))))
        #validation_transforms.append(A.augmentations.transforms.RandomGamma(p=augmentation_params.get('gamma_p', 0.5)))

    data_transforms = {
        'train': A.Compose(train_transforms),
        'validation': A.Compose(validation_transforms)
    }

    return data_transforms

def get_model_config(config: dict):
    return VideoMAEConfig(**config)


def get_pretraining_model(config):
    mae_config = get_model_config(config['model'])
    return CustomVideoMAEForPretraining(mae_config)

def get_regressor_model(config):
    mae_config = get_model_config(config['model'])
    dropout = config['model'].get('dropout', 0.2)
    return CustomVideoMAEForRegression(mae_config, dropout=dropout)


def get_activation(activation_name):
    '''Returns an activation function based on the activation_name

    Parameters
    ----------
    activation_name : str
        Name of the activation function
    
    Returns
    -------
    torch.nn.Module
        Activation function'''
    activation_name = str(activation_name)
    if activation_name == 'ReLU':
        return torch.nn.ReLU
    elif activation_name == 'LeakyReLU':
        return torch.nn.LeakyReLU
    elif activation_name == 'ELU':
        return torch.nn.ELU
    elif activation_name == 'GELU':
        return torch.nn.GELU
    elif activation_name == 'SELU':
        return torch.nn.SELU
    else:
        raise ValueError(f'Activation type not supported: {activation_name}')

def get_optimizer(optimizer_params, model, separate_params=False, **kwargs):
    '''Returns an optimizer based on the optimizer_params
    
    Parameters
    ----------
    optimizer_params : dict
        Dictionary containing the optimizer parameters
    model : torch.nn.Module
        Model to be trained
        
    Returns
    -------
    torch.optim.Optimizer
        Optimizer to be used for training'''
    optimizer_type = optimizer_params['optimizer']
    learning_rate = optimizer_params['learning_rate']
    if optimizer_type == 'Adam':
        if separate_params:
            return torch.optim.Adam([
                {'params': model.model.videomae.parameters(), 'lr': learning_rate*0.1},
                {'params': model.model.lvef_head.parameters(), 'lr': learning_rate},
                {'params': model.model.rvef_head.parameters(), 'lr': learning_rate}
            ], lr=learning_rate, **kwargs)
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError('Optimizer type not supported')
    
def get_scheduler(train_params, optimizer):
    '''Returns a scheduler based on the train_params
    
    Parameters
    ----------
    train_params : dict
        Dictionary containing the train parameters
    optimizer : torch.optim.Optimizer
        Optimizer to be used for training
    
    Returns
    -------
    dict
        Dictionary containing the optimizer, scheduler and its parameters'''
    scheduler_type = train_params['scheduler']
    scheduler_params = train_params.get(f'{scheduler_type}_params', {})
    if scheduler_type is None:
        return None
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params.get('step_size', 5),
                                                    gamma=scheduler_params.get('gamma', 0.1))
    elif scheduler_type == 'PlateauLR':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode=scheduler_params.get('mode', 'min'),
                                                          factor=scheduler_params.get('factor', 0.1),
                                                          patience=scheduler_params.get('patience', 5),
                                                          verbose=True)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=scheduler_params.get('T_max', 10),
                                                          eta_min=scheduler_params.get('eta_min', 0))
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=scheduler_params.get('T_0', 10),
                                                                    T_mult=scheduler_params.get('T_mult', 1),
                                                                    eta_min=scheduler_params.get('eta_min', 0))
    else:
        raise ValueError('Scheduler type not supported')
    
    return {'scheduler': scheduler, 'monitor': 'val_loss', 'mode': 'min', 'interval': 'epoch', 'frequency': 1}

def get_callbacks(callback_params):
    '''Returns a list of callbacks based on the callback_params dictionary
    
    Parameters
    ----------
    callback_params : dict
        Dictionary containing the parameters for the callbacks
        
    Returns
    -------
    list
        List of callbacks
    '''
    callbacks = []
    if 'model_checkpoint' in callback_params.keys():
        mc_params = callback_params['model_checkpoint']
        callbacks.append(ModelCheckpoint(monitor=mc_params.get('monitor', 'val_mae_patient'),
                                         filename='epoch={epoch}-{train_loss:.2f}',
                                         mode=mc_params.get('mode', 'min'),
                                         save_top_k=mc_params.get('save_top_k', 1),
                                         save_last=True,
                                         verbose=mc_params.get('verbose', True),))
    return callbacks

def get_weight_function(train_dataset, alpha=0.5):
    y_true = np.array([np.mean(train_dataset.get_label(idx)) for idx in range(len(train_dataset))]).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(y_true)
    mms = MinMaxScaler()
    dens_norm = mms.fit_transform(np.exp(kde.score_samples(y_true)).reshape(-1,1))
    weights = 1 - alpha*dens_norm
    def calculate_weight(y):
        device = y.get_device()
        y = y.detach().cpu().reshape(-1,1)
        dens_y = mms.transform(np.exp(kde.score_samples(y)).reshape(-1,1))
        weight_y = 1 - alpha*dens_y
        return  torch.tensor((weight_y / (np.mean(weights))), requires_grad=False).to(device)
    return calculate_weight

def get_weighted_mse(weight_function):
    def weighted_mse(input, target):
        return (weight_function(target) * (target - input)**2).mean()
    return weighted_mse

def get_weighted_mae(weight_function):
    def weighted_mae(input, target):
        return (weight_function(target) * torch.abs(target - input)).mean()
    return weighted_mae