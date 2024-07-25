import torch
from torch import optim, nn


def create_optimizer(config, model):
    lr = config['lr']
    weight_decay = config.get('weight_decay', 0)
    betas = tuple(config.get('betas', (0.9, 0.999)))
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    return optimizer

def create_loss(config):
    loss_name = config.pop('name')
    if loss_name == 'CrossEntropyLoss':
        weight = config.get('weight', None)
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_name == 'L1Loss':
        return nn.L1Loss()
    else:
        raise ValueError(f"unsupported loss function: '{loss_name}'")

