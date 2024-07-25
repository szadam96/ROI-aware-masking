from torch.utils.data import DataLoader, RandomSampler
from dataset.video_dataset import EchoVideoDataset
from dataset.data_sampler import EchoDataSampler
import torch
import pickle
import numpy as np
from pathlib import Path
import os
import lightning as pl
from utils.train_utils import get_augmentations


class EchoVideoDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, root_dir, augmentations,
                 test_path=None,
                 batch_size=10, num_workers=8, balance=True,
                 include_binary_mask=False, to_predict=['LVEF', 'RVEF'], **kwargs):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance = balance
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        elif isinstance(root_dir, list):
            root_dir = [Path(r) for r in root_dir]
        self.root_dir = root_dir
        self.transform = get_augmentations(augmentations)
        self.include_binary_mask = include_binary_mask
        self.to_predict = to_predict
        self.sampler_kwargs = kwargs

    def setup(self, stage=None):
        self.train_data = EchoVideoDataset(self.train_path, root_dir=self.root_dir,
                                           transform=self.transform['train'],
                                           include_binary_mask=self.include_binary_mask,
                                           to_predict=self.to_predict)
        self.val_data = EchoVideoDataset(self.val_path, root_dir=self.root_dir,
                                         transform=self.transform['validation'],
                                         include_binary_mask=self.include_binary_mask,
                                         to_predict=self.to_predict)
        if self.test_path is not None:
            self.test_data = EchoVideoDataset(self.test_path, root_dir=self.root_dir,
                                              transform=self.transform['validation'],
                                              include_binary_mask=self.include_binary_mask,
                                              to_predict=self.to_predict)

        if self.balance:
            self.train_sampler = EchoDataSampler(self.train_data, balance=self.balance,
                                                 **self.sampler_kwargs)
        else:
            self.train_sampler = RandomSampler(self.train_data)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=self.train_sampler,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)