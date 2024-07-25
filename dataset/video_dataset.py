from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class EchoVideoDatasetBase(Dataset):
    def __init__(self, json_file,
                 root_dir,
                 transform,
                 mask_file_name='mask.png',
                 include_binary_mask=False):
        with open(json_file, 'r') as data:
            json_data = json.load(data)
        self.data = json_data

        self.root_dir = root_dir
        self.transform = transform
        self.mask_file_name = mask_file_name
        self.frame_list = self.create_frame_list()
        self.include_binary_mask = include_binary_mask

    def create_frame_list(self):
        raise NotImplementedError('create_frame_list not implemented')

    def apply_transform(self, frames, mask):
        targets = {f'frame{i}': 'image' for i in range(len(frames))}
        pars = {f'frame{i}': frame for i, frame in enumerate(frames)}
        pars['image'] = frames[0]
        pars['mask'] = mask
        self.transform.add_targets(targets)
        transformed = self.transform(**pars)

        transformed_frames = np.asarray([transformed[f'frame{i}'] for i in range(frames.shape[0])])
        transformed_mask = transformed['mask']

        return transformed_frames, transformed_mask
    
    def to_tensor(self, frames, mask):
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32)
            frames /= 255.0

        frames = torch.from_numpy(frames).float()

        return frames
    
    def __len__(self):
        return len(self.frame_list)
    
    def load_frames_and_mask(self, frames_path, frame_indexes):
        frames = []
        mask_path = frames_path.parent / self.mask_file_name
        for i in frame_indexes:
            frame = Image.open(frames_path / f'frame{i}.png')
            frame = np.asarray(frame)
            frames.append(frame)

        try:
            frames = np.asarray(frames)
            frames = np.transpose(frames, (0, 1, 2))
        except ValueError as e:
            print(f'ValueError: {frame_indexes}\n {frames_path}')
            for f in frames:
                print(f[0].shape)
            raise e
        binary_mask = np.asarray(Image.open(mask_path))/255.0
        
        return frames, binary_mask
    
    def __getitem__(self, idx):
        raise NotImplementedError('__getitem__ not implemented')




class EchoVideoDataset(EchoVideoDatasetBase):
    def __init__(self, json_file,
                 root_dir,
                 transform,
                 mask_file_name='mask.png',
                 include_binary_mask=False,
                 to_predict=['RVEF', 'LVEF']):
        self.to_predict = to_predict
        super().__init__(json_file,
                         root_dir,
                         transform,
                         mask_file_name=mask_file_name,
                         include_binary_mask=include_binary_mask
                         )

    def create_frame_list(self):
        key_list = []
        self.id_list = []
        for patient_id, val in self.data.items():
            lvef = val.get('LVEF', np.nan)
            for dicom in val['dicoms']:
                orientation = dicom['orientation'].lower()
                dicom_id = dicom['dicom_id']
                if patient_id == '':
                    dicom_folder = dicom_id
                else:
                    dicom_folder = f'{patient_id}_{dicom_id}'
                if isinstance(dicom_id, int):
                    dicom_id = f'{dicom_id:04d}'
                for frame_indexes in dicom['frame_indexes']:
                    if isinstance(self.root_dir, Path):
                        frames_path = self.root_dir / dicom_folder / 'frames'
                    elif isinstance(self.root_dir, list):
                        for root in self.root_dir:
                            frames_path = root / dicom_folder / 'frames'
                            if frames_path.exists():
                                break
                    key_list.append((patient_id, dicom_id, frames_path, frame_indexes, orientation, lvef))
        return key_list

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        patient_id, dicom_id, frames_path, frame_indexes, orientation, lvef = self.frame_list[idx]

        frames, binary_mask = self.load_frames_and_mask(frames_path, frame_indexes)
        
        if orientation == 'stanford':
            frames = np.flip(frames, axis=2).copy()
            binary_mask = np.flip(binary_mask, axis=1).copy()

        if self.transform:
            frames, binary_mask = self.apply_transform(frames, binary_mask)

        input_tensor = self.to_tensor(frames, binary_mask)

        sample = {'input_tensor': input_tensor.unsqueeze(1), 'binary_mask': binary_mask, 'LVEF': lvef, 'patient_id': patient_id, 'dicom_id': dicom_id}
                
        return sample

    def get_label(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        res = []  
        if 'RVEF' in self.to_predict:
            res.append(self.frame_list[idx][-2])
        if 'LVEF' in self.to_predict:
            res.append(self.frame_list[idx][-1])
        
        if res == []:
            raise ValueError('No label to predict')
        return res