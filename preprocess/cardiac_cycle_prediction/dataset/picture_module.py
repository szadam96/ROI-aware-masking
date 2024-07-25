from typing import Dict

import torch
from torch.utils import data
import pytorch_lightning as pl


class Echo2dPicturesDM(pl.LightningDataModule):
    def __init__(self, datasets: Dict, batch_size=1, num_workers=1):
        super().__init__()
         
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return data.DataLoader(
            self.datasets['train'],
            #data.TensorDataset(torch.rand(1000, 1, 224, 224), torch.randint(2, size=(1000,))),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )   
    
    def val_dataloader(self):
        return data.DataLoader(
            self.datasets['val'],
            #data.TensorDataset(torch.rand(100, 1, 224, 224), torch.randint(2, size=(100,))),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )   
    
    def test_dataloader(self):
        return data.DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def predict_dataloader(self):
        return data.DataLoader(
            self.datasets['predict'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
