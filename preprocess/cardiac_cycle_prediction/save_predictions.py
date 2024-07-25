import yaml
import torch
import pytorch_lightning as pl
from preprocess.cardiac_cycle_prediction.dataset.utils import create_data_module
from preprocess.cardiac_cycle_prediction.model.model import create_model_module
from preprocess.cardiac_cycle_prediction.utils.callbacks import PredictionWriter

def load_config(config_path):
    with open(str(config_path), 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_predictions(model_config, output_config, data_csv, devices=1):
    model_config['data_module']['datasets']['data_csv'] = data_csv
    model_config['data_module']['datasets']['predict']['input_dir'] = output_config['data_dir']

    data_module = create_data_module(model_config['data_module'])
    model_module = create_model_module(model_config['model_module'])
    writer = PredictionWriter(output_config['data_dir'])

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        callbacks=[writer],
    )

    model_module.eval()
    trainer.predict(model_module, data_module, ckpt_path=output_config['cardiac_cycle_prediction']['checkpoint_path'])
