import os
from collections import defaultdict
from pathlib import Path
import pandas as pd

from torchvision import transforms
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from preprocess.cardiac_cycle_prediction.utils.utils import get_class


def read_orientation_file(path):
    orientation_data = dict()

    df = pd.read_csv(path)
    #df.dropna(subset=['orientation'], inplace=True)
    for idx, row in df.iterrows():
        dicom_id = row['dicom_id']
        try:
            orientation = row['orientation'].lower()
        except KeyError:
            orientation = 'mayo'
        orientation_data[dicom_id] = orientation

    return orientation_data

def extract_idx_from_name(name):
    stem = Path(name).stem
    assert stem[:5] == 'frame'
    idx_str = stem[5:]
    return int(idx_str)


def create_data_module(config):
    name = config['name']
    batch_size = config.get('batch_size', 1)
    num_workers = config.get('num_workers', 1)
    datasets_config = config['datasets']
    datasets = create_datasets(datasets_config)

    clazz = get_class(name, modules=['cardiac_cycle_prediction.dataset.picture_module'])
    data_module = clazz(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return data_module

def create_datasets(config):
    annotation_path = config['data_csv'] if 'data_csv' in config else None
    orientation_data = None
    if annotation_path is not None:
        orientation_data = read_orientation_file(annotation_path)

    datasets = dict()
    for split in config:
        if split != 'data_csv':
            split_dataset = create_split_dataset(config[split], None, orientation_data)
            datasets[split] = split_dataset

    return datasets
    

def create_split_dataset(config, annotation_data, orientation_data):
    name = config['name']
    input_dir = config['input_dir']
    create_ds_fn = create_echo_video_dataset

    datasets = []
    print(f'Creating dataset from {input_dir}')
    for dicom_id in tqdm(os.listdir(input_dir)):
        ds = create_ds_fn(config, annotation_data, orientation_data, dicom_id, input_dir)
        datasets.append(ds)

    return ConcatDataset(datasets)

def create_regular_dataset(config, annotation_data=None, orientation_data=None, dicom_id=None, input_dir=None):
    name = config['name']
    preprocessor = None
    transform = create_transform(None)
    clazz = get_class(name, modules=['cardiac_cycle_prediction.dataset.picture_dataset'])
    input_folder = os.path.join(input_dir, dicom_id)

    ds = clazz(
        input_folder=input_folder,
        orientation=orientation_data[dicom_id] if orientation_data is not None else 'Mayo',
        transform=transform,
        preprocessor=preprocessor
    )

    return ds

def create_echo_video_dataset(config, annotation_data, orientation_data, dicom_id, input_dir):
    frames_per_video = config.get('frames_per_video', None)
    seed = config.get('seed', 1043)
    random_start = config.get('random_start', True)
    pictures_ds_config = config['pictures_dataset']
    pictures_ds = create_regular_dataset(pictures_ds_config, annotation_data, orientation_data, dicom_id, input_dir)
   
    # TODO: fix this (circular import workaround)
    clazz = get_class('Echo2dVideo', modules=['cardiac_cycle_prediction.dataset.picture_dataset'])
    ds = clazz(
        pictures_ds,
        frames_per_video=frames_per_video,
        seed=seed,
        random_start=random_start
    )
    return ds

def create_transform(config):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform

