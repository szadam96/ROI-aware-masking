# Masked autoencoders for medical ultrasound videos using ROI-aware masking

This is the implementation of MICCAI-ASMUS 24 submission titled "Masked autoencoders for medical ultrasound videos using ROI-aware masking".

## Introduction

The code is structured as follows:
- `preprocess`: Contains the code for preprocessing the data.
- `pretraining`: Contains the code for pretraining the network.
- `training`: Contains the code for training the network.

## Requirements

The code is tested on Python 3.9 and PyTorch 2.0. The full list of dependencies is available in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Datasets

The dataset used for the preptraining in the paper are not publicly available. However, the code can be used with other datasets, with 2D cardiac ultrasound videos that are in a standard DICOM format. The dataset for the suoervised training is available at https://rvenet.github.io/dataset/.

## Preprocess

The preprocess step generates the data needed for the pretraining and training steps.

### Usage

```
python preprocess.py (--path_to_data PATH_TO_DATA | --path_to_csv PATH_TO_CSV) --config config/preprocess.yaml [--orientation {mayo,stanford}]
```

The configuration file `config/preprocess.yaml` contains all the parameters for the preprocess step.
The `--path_to_data` argument is used to specify the path to the directory containing the videos.
The `--orientation` argument is used to specify the orientation of the videos. The default value is `mayo`.
Alternatively a csv file containing the path of each video can be provided with the `--path_to_csv` flag. The table needs to inclide a column named `dicom_path`. In addition, a unique label and orientation can also be provided in the csv for each video. The name of this label must be provided in the config file.

## Pretraining


### Usage

```bash
python train_self_supervised.py --config config/pretraining.yaml --log_path logs/
```

The configuration file `config/pretraining.yaml` contains all the parameters for the pretraining step.The `--log_path` argument is used to specify the path to the directory where the logs will be saved.

## Training

### Usage

```bash
python train_supervised.py --config config/training.yaml --log_path logs/
```

The configuration file `config/training.yaml` contains all the parameters for the training step.The `--log_path` argument is used to specify the path to the directory where the logs will be saved.