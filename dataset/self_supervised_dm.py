from torch.utils.data import DataLoader, RandomSampler
from dataset.video_dataset_pretraining import EchoVideoDataset
import numpy as np
from pathlib import Path
import os
import lightning as pl
from utils.train_utils import get_augmentations, get_model_config


class EchoVideoDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, root_dir, augmentations, mae_config,
                 batch_size=10, num_workers=8, balance=True,
                 roi_aware=False, masking_precentage=0.5, **kwargs):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance = balance
        self.root_dir = root_dir
        self.transform = get_augmentations(augmentations)
        self.include_binary_mask = roi_aware
        self.masking_precentage = masking_precentage
        self.sampler_kwargs = kwargs
        self.mae_config = get_model_config(mae_config)

    def setup(self, stage=None):
        self.train_data = EchoVideoDataset(self.train_path,
                                           root_dir=Path(self.root_dir),
                                           mae_config=self.mae_config,
                                           transform=self.transform['train'],
                                           include_binary_mask=self.include_binary_mask,
                                           masking_precentage=self.masking_precentage)
        self.val_data = EchoVideoDataset(self.val_path,
                                         root_dir=Path(self.root_dir),
                                         mae_config=self.mae_config,
                                         transform=self.transform['validation'],
                                         include_binary_mask=self.include_binary_mask,
                                         masking_precentage=self.masking_precentage)
        
        self.train_sampler = RandomSampler(self.train_data, num_samples=self.sampler_kwargs.get('num_samples', None))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=self.train_sampler,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)
