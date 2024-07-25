import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataset.video_dataset import EchoVideoDatasetBase
import numpy as np
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class EchoVideoDataset(EchoVideoDatasetBase):
    def __init__(self, json_file,
                 root_dir,
                 transform,
                 mae_config,
                 mask_file_name='mask.png',
                 masking_precentage=0.75,
                 include_binary_mask=False):
        super().__init__(json_file,
                         root_dir,
                         transform,
                         mask_file_name,
                         include_binary_mask)
        self.mae_config = mae_config
        self.masking_precentage = masking_precentage

    def create_frame_list(self):
        key_list = []
        self.id_list = []
        for dicom_id, val in self.data.items():
            for frame_indexes in val['frame_indexes']:
                frames_path = self.root_dir / dicom_id / 'frames'
                try:
                    orientation = val['orientation']
                except KeyError:
                    orientation = 'mayo'
                key_list.append((dicom_id, frames_path, frame_indexes, orientation))
        return key_list

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        dicom_id, frames_path, frame_indexes, orientation = self.frame_list[idx]

        frames, binary_mask = self.load_frames_and_mask(frames_path, frame_indexes)
        
        if self.transform:
            frames, binary_mask = self.apply_transform(frames, binary_mask)

        if orientation == 'stanford':
            frames = np.flip(frames, axis=2).copy()
            binary_mask = np.flip(binary_mask, axis=1).copy()

        if not self.include_binary_mask:
            binary_mask = None

        bool_masked_pos = self.create_masked_pos(self.mae_config, binary_mask, self.masking_precentage)

        input_tensor = self.to_tensor(frames, binary_mask)

        sample = {'input_tensor': input_tensor.unsqueeze(1), 'bool_masked_pos': bool_masked_pos, 'binary_mask': binary_mask, 'dicom_id': dicom_id}
                
        return sample
    
    def create_masked_pos(self, mae_config, mask=None, masking_percentage=0.5):
        if mask is not None:
            mask = ~F.max_pool2d(torch.tensor(mask.copy()).unsqueeze(0).unsqueeze(0).float(), mae_config.patch_size).squeeze().bool()
        else:
            mask = torch.zeros((mae_config.image_size // mae_config.patch_size, mae_config.image_size // mae_config.patch_size)).bool()

        num_elements = mask.numel()
        num_false_elements = (~mask).sum().item()
        num_unmasked_elements = int((1-masking_percentage) * num_elements)
        if num_unmasked_elements > num_false_elements:
            mask = torch.zeros((mae_config.image_size // mae_config.patch_size, mae_config.image_size // mae_config.patch_size)).bool()
            num_false_elements = (~mask).sum().item()

        indices = torch.nonzero(~mask)

        masked_indices = indices[torch.randperm(num_false_elements)[:num_unmasked_elements]]

        mask = ~torch.zeros((mae_config.image_size // mae_config.patch_size, mae_config.image_size // mae_config.patch_size)).bool()

        mask[masked_indices[:, 0], masked_indices[:, 1]] = False


        mask = torch.stack([mask for _ in range(mae_config.num_frames//mae_config.tubelet_size)], dim=0).flatten()
        
        return mask