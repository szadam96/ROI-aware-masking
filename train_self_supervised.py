from argparse import ArgumentParser
import os
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.self_supervised_dm import EchoVideoDataModule
from models.model_pretraining import EchoModel
from utils.train_utils import get_callbacks
from utils.utils import fix_seed

import torch.nn as nn
import torch.nn.functional as F


def run_training(para_path=None, log_path='logs'):
    '''Run training for the model
    
    Parameters
    ----------
    para_path : str
        Path to the yaml file containing the parameters
    '''

    # Load augmentation yaml
    with open(para_path, 'r') as f:
        parameters = yaml.safe_load(f)

    # Set seed
    fix_seed(parameters['seed'])

    augmentations = parameters['augmentations']
    dataloader_params = parameters['dataloader']

    train_path = dataloader_params['train_json']
    val_path = dataloader_params['val_json']

    dm = EchoVideoDataModule(train_path, val_path,
                             augmentations=augmentations,
                             mae_config=parameters['model'],
                             **dataloader_params)

    tensorboard_logger = TensorBoardLogger(log_path)

    # Initialize model
    model = EchoModel(parameters)

    # Initialize callback
    callbacks = get_callbacks(parameters['callbacks'])

    # Initialize trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=2,
                         max_epochs=parameters['epochs'],
                         logger=tensorboard_logger,
                         callbacks=callbacks,
                         precision=parameters['precision'])

    # Train
    trainer.fit(model, dm)

def main():
    args = ArgumentParser()
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--log_path', type=str, default='logs')
    args = args.parse_args()
    run_training(para_path=args.config, log_path=args.log_path)


if __name__ == '__main__':
    main()
