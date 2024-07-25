from pathlib import Path
from dataset.video_dataset import EchoVideoDataset
from models.model import EchoModel
from utils.train_utils import get_augmentations, get_callbacks
from utils.utils import fix_seed
from utils.prediction_writer import PredictionWriter
import torch
import lightning as pl
from lightning.pytorch import loggers as pl_loggers
import tensorboardX
from torch.utils.data import DataLoader
from argparse import ArgumentParser


def run_prediction(model_path, root_dir, data_json, output_dir):
    '''
    Run prediction for the model

    Parameters
    ----------
    model_path : str
        Path to the checkpoint
    root_dir : str
        Path to the preprocessed DICOM files
    data_json : str
        Path to the JSON file containing the dataset
    output_dir : str
        Path to the output directory
    '''
    ckpt = torch.load(model_path, map_location='cpu')
    parameters = ckpt['hyper_parameters']
    
    # Set seed
    fix_seed(parameters['seed'])

    parameters['model']['ckpt_path'] = None

    augmentations = parameters['augmentations']
    dataloader_params = parameters['dataloader']

    transform = get_augmentations(augmentations)

    data_set = EchoVideoDataset(data_json,
                                root_dir=Path(root_dir),
                                transform=transform['validation'],
                                include_binary_mask=dataloader_params['include_binary_mask'])

    dataloader = DataLoader(data_set, batch_size=dataloader_params['batch_size'], num_workers=8)

    tensorboard_logger = pl_loggers.TensorBoardLogger(output_dir)
    
    # Initialize model
    model = EchoModel(parameters)
    model.load_state_dict(ckpt['state_dict'])

    callbacks = [PredictionWriter(output_dir=output_dir)]

    # Initialize trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=parameters['epochs'],
                         logger=tensorboard_logger,
                         callbacks=callbacks,
                         precision=parameters['precision'])

    # Train
    trainer.predict(model, dataloader)


def main():
    args = ArgumentParser()
    args.add_argument('--model_path', type=str, required=True)
    args.add_argument('--root_dir', type=str, required=True)
    args.add_argument('--data_json', type=str, required=True)
    args.add_argument('--output_dir', type=str, required=True)

    args = args.parse_args()

    run_prediction(args.model_path, args.root_dir, args.data_json, args.output_dir)


if __name__ == '__main__':
    main()
