import os
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
import pandas as pd

from torch.utils.data.dataset import Dataset
from torch.utils.data import default_collate

from .utils import extract_idx_from_name
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AbstractEcho2dPictures(Dataset):
    def __init__(
            self,
            input_folder,
            orientation,
            transform=None,
            internal_path='frames',
            **kwargs
    ):
        self.input_folder = input_folder
        self.internal_path = internal_path
        self.images_folder = os.path.join(input_folder, internal_path)
        self.dicom_id = os.path.basename(self.input_folder) #.split('_', 1)[1]

        orientation_lower = orientation.lower()
        assert orientation_lower in ['mayo', 'stanford'], f'received unsupported orientation: {orientation}'
        self.orientation = orientation_lower

        self.transform = transform
        
        sample_names = os.listdir(self.images_folder)
        self.sorted_sample_names = sorted(sample_names, key=extract_idx_from_name)

    def __len__(self):
        return len(self.sorted_sample_names)

    def __getitem__(self, idx):
        sample_name = self.sorted_sample_names[idx]
        sample_path = os.path.join(self.images_folder, sample_name)
        sample = Image.open(sample_path)

        sample = self._preprocess_image(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        label = self._get_label(idx)
    
        return {
            'input': sample,
            'target': label,
            'dicom_id': self.dicom_id,
            'filename': sample_name
        }

    
    def _preprocess_image(self, image):
        if self.orientation == 'stanford':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def _get_label(self, idx):
        raise NotImplementedError

class Echo2dPicturesPredict(AbstractEcho2dPictures):
    def __init__(self, *args, **kwargs):
        super(Echo2dPicturesPredict, self).__init__(*args, **kwargs)

    def _get_label(self, idx):
        return 0

class Echo2dVideo(Dataset):
    def __init__(self, pictures_dataset, frames_per_video=None, seed=1043, random_start=True):
        super(Echo2dVideo, self).__init__()
        self.pictures_dataset = pictures_dataset
        self.frames_per_video = frames_per_video
        self.seed = seed
        self.random_state = np.random.RandomState(seed=seed)
        self.random_start = random_start

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.frames_per_video is not None:
            n_frames = len(self.pictures_dataset)
            if n_frames <= self.frames_per_video:
                frames_data = self._self_padded_frames()
            else:
                start_idx = self._get_start_idx()
                frames_data = [self.pictures_dataset[idx] for idx in range(start_idx, start_idx+self.frames_per_video)]
        else:
            frames_data = [sample for sample in self.pictures_dataset]

        frames_data = default_collate(frames_data)
        return frames_data

    def _get_start_idx(self):
        if self.random_start:
            n_frames = len(self.pictures_dataset)
            n_start_indices = n_frames - self.frames_per_video + 1
            start_idx = self.random_state.randint(0, n_start_indices)
        else:
            start_idx = 0
        return start_idx

    def _self_padded_frames(self):
        n_pictures = len(self.pictures_dataset)
        k = int(np.ceil(self.frames_per_video / n_pictures))
        frame_indices = list(range(n_pictures))
        padded_indices = np.tile(frame_indices, k)
        padded_indices = padded_indices[:self.frames_per_video]
        
        assert len(padded_indices) == self.frames_per_video

        return [self.pictures_dataset[idx] for idx in padded_indices]

    
