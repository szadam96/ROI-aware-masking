from argparse import ArgumentParser
from dataset.data_module import EchoVideoDataModule
from models.model import EchoModel
from utils.train_utils import get_callbacks, get_weight_function
from utils.utils import fix_seed

import yaml

import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch import loggers as pl_loggers


def run_training(para_path=None,
                 log_path='logs'):
    '''Run training for the model
    
    Parameters
    ----------
    para_path : str
        Path to the yaml file containing the parameters
    log_path : str
        Path to the log directory
    '''
    with open(para_path, 'r') as f:
        parameters = yaml.safe_load(f)
    
    # Set seed
    fix_seed(parameters['seed'])

    augmentations = parameters['augmentations']

    parameters['train_path'] = train_path
    

    dataloader_params = parameters['dataloader']
    dataloader_params['to_predict'] = parameters['training']['to_predict']

    train_path = dataloader_params['train_json']
    val_path = dataloader_params['val_json']
    test_path = dataloader_params['test_json']

    dm = EchoVideoDataModule(train_path, val_path, test_path=test_path,
                             augmentations=augmentations,
                             mae_config=parameters['model'],
                             **dataloader_params)
    
    # Set up weight function for weighted loss
    weight_func = None
    if parameters['training']['weighted']:
        dm.setup()
        weight_func = get_weight_function(dm.train_data, parameters['training']['weight_alpha'])

    tensorboard_logger = pl_loggers.TensorBoardLogger(log_path)

    model = EchoModel(parameters, weight_func=weight_func)

    #model = torch.compile(model)

    # Initialize callback
    callbacks = get_callbacks(parameters['callbacks'])

    # Initialize trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=[1],
                         max_epochs=parameters['epochs'],
                         logger=tensorboard_logger,
                         callbacks=callbacks,
                         precision=parameters['precision'])

    # Train
    trainer.fit(model, dm)

    # Test
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.test(model, datamodule=dm, ckpt_path=best_ckpt_path)

def main():
    args = ArgumentParser()
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--log_path', type=str, required=True)
    args.parse_args()
    run_training(para_path=args.config, log_path=args.log_path)

if __name__ == '__main__':
    main()