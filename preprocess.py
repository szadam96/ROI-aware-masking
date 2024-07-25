from argparse import ArgumentParser
import os
from pathlib import Path

import pandas as pd

from preprocess.save_frames import find_dicom_files, preprocess_multiprocess
from preprocess.cardiac_cycle_prediction.save_predictions import load_config, save_predictions
from preprocess.create_json import create_jsons

def process_dicoms(path_to_csv: Path, output_folder: Path, skip_saving: bool = False):
    '''
    Preprocess the DICOM files and save them to the output folder.
    
    Parameters
    ----------
    path_to_csv : Path
        The path to the CSV containing the paths to the DICOM files.
    output_folder : Path
        The path to the folder where the preprocessed DICOM files will be saved.
    skip_saving : bool
        Whether to skip saving the preprocessed DICOM files.
    '''
    df = pd.read_csv(path_to_csv)
    preprocess_multiprocess(df, output_folder, path_to_csv, skip_saving)

def run_cardiac_cycle_prediction(config: dict, data_csv: Path, model_config_path: Path):
    '''
    Run the cardiac cycle prediction model on the preprocessed DICOM files.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    data_csv : Path
        The path to the CSV containing the paths to the DICOM files.
    model_config_path : Path
        The path to the model configuration file.
    '''
    model_config = load_config(model_config_path)
    save_predictions(model_config=model_config,
                     output_config=config,
                     data_csv=data_csv)
    
def create_dataset_jsons(config: dict, data_csv: Path):
    '''
    Create the JSON files for the dataset.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    data_csv : Path
        The path to the CSV containing the paths to the DICOM files.
    '''
    output_path = config['json_generation']['output']
    frames_to_sample = config['json_generation']['frames_to_sample']
    label = config['json_generation']['label']
    name = config['json_generation']['name']
    create_jsons(path_to_csv=data_csv,
                    output_path=Path(output_path),
                    path_to_preprocessed_dicoms=Path(config['data_dir']),
                    frames_to_sample=frames_to_sample,
                    label=label,
                    name=name)

def add_orientation(data_csv: Path, orientation: str):
    '''
    Add the orientation to the CSV containing the paths to the DICOM files if it is not already present.

    Parameters
    ----------
    data_csv : Path
        The path to the CSV containing the paths to the DICOM files.
    orientation : str
        The orientation of the A4C DICOM files in the dataset.
    '''
    df = pd.read_csv(data_csv)
    if 'orientation' not in df.columns:
        df['orientation'] = orientation
    df.to_csv(data_csv, index=False)

def main():
    parser = ArgumentParser()
    group = parser.add_argument_group('Input data',
                                      'Provide the path to the folder containing all DICOM files or'
                                      'a CSV file containing the paths to the DICOM files to be analyzed.')
    input_group = group.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--path_to_data', help='The path to the folder containing the DICOM files.')
    input_group.add_argument('--path_to_csv', help='The path to the CSV containing the paths to the DICOM files.')
    parser.add_argument('--config', type=Path, default='config.yaml')
    parser.add_argument('--orientation', type=str, default='mayo', required=False, help='The orientation of the A4C DICOM files in the dataset.', choices=['mayo', 'stanford'])

    args = parser.parse_args()
    if args.path_to_data:
        dicom_files_list = find_dicom_files(args.path_to_data)
        df = pd.DataFrame(dicom_files_list, columns=['dicom_path'])
        csv_path = 'dicom_files.csv'
        df.to_csv(csv_path, index=False)
        data_csv = csv_path
    else:
        data_csv = args.path_to_csv
    
    config = load_config(args.config)
    cc_prediction_config = 'preprocess/cardiac_cycle_prediction/model_config.yaml'

    process_dicoms(data_csv, Path(config['data_dir']))
    add_orientation(data_csv, args.orientation)
    run_cardiac_cycle_prediction(config, data_csv, cc_prediction_config)
    create_dataset_jsons(config, data_csv)

if __name__ == '__main__':
    main()