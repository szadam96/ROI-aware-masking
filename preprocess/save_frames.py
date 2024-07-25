import argparse
import multiprocessing
import os
from os.path import exists
import cv2
import pydicom
from pydicom.pixel_data_handlers import convert_color_space
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import warnings
from matplotlib import pyplot as plt


warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def is_3d_dicom(dcm):
    vendor = dcm.Manufacturer
    if 'GE' in vendor:
        try:
            tag = dcm[0x7fe1, 0x1001][0][0x7fe1, 0x1002].value
        except:
            return False
        if '3D' in tag:
            return True
        return False
    elif 'Philips' in vendor:
        try:
            dcm[0x200d, 0x3cf5][1][0x200d, 0x3cf1][0][0x200d, 0x3cf3]
        except:
            return False
        return True
    
    return False



def bounding_box(points):
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)

    return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}


def create_mask(gray_frames, reference_pixel):
    # Computing the extent and frequency of changes in pixel intensity values
    changes = np.zeros((gray_frames.shape[1], gray_frames.shape[2]))
    changes_frequency = np.zeros((gray_frames.shape[1], gray_frames.shape[2]))
    binary_mask = np.zeros((gray_frames.shape[1], gray_frames.shape[2]))

    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i].astype(np.int32) - gray_frames[i + 1].astype(np.int32))
        diff[diff < 5] = 0
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1

    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < round(gray_frames.shape[0] * 0.1):
                changes[r][p] = 0

    nonzero_values_for_binary_mask = np.nonzero(changes)

    # Generating a binary mask based on the changes in pixel intensity values
    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1

    # Opening
    kernel = np.ones((5, 5), np.int32)
    opening_on_binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = np.where(opening_on_binary_mask, binary_mask, 0)

    # Closing
    closing_on_binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = np.where(closing_on_binary_mask, binary_mask, 0)

    # Finding the contour with the largest area
    contours, hierarchy = cv2.findContours(binary_mask.astype(np.uint8),
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_mask = np.zeros(binary_mask.shape)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 1, -1)

    # Finding the convex hull of the points of the largest contour
    if reference_pixel:
        largest_contour = np.append(largest_contour, [[reference_pixel]], axis=0)
    hull = cv2.convexHull(largest_contour)
    hull_mask = np.zeros(binary_mask.shape)
    cv2.drawContours(hull_mask, [hull], -1, 1, -1)

    # Finding the bounding box of the final mask
    nonzero_values_hull_mask = np.nonzero(hull_mask)
    hull_mask_coordinates = np.array([nonzero_values_hull_mask[0], nonzero_values_hull_mask[1]]).T
    bbox = bounding_box(hull_mask_coordinates)

    # Cropping the masks
    cropped_binary_mask = binary_mask[bbox['min_x']:bbox['max_x'], bbox['min_y']:bbox['max_y']]
    cropped_largest_contour_mask = largest_contour_mask[bbox['min_x']:bbox['max_x'], bbox['min_y']:bbox['max_y']]
    cropped_hull_mask = hull_mask[bbox['min_x']:bbox['max_x'], bbox['min_y']:bbox['max_y']]

    return cropped_binary_mask, cropped_largest_contour_mask, cropped_hull_mask, bbox


def save_frames(dicom_dataset, path_to_save, frame_limit=None, flip=False):
    # Extracting the pixel array from the DICOM dataset
    pixel_array = dicom_dataset.pixel_array

    # Convert color space (if necessary)
    if hasattr(dicom_dataset, 'PhotometricInterpretation'):
        if dicom_dataset.PhotometricInterpretation == 'RGB':
            pixel_array = convert_color_space(pixel_array, 'RGB', 'YBR_FULL_422', per_frame=True)

    # Checking whether the video has enough frames for analysis
    if pixel_array.shape[0] < 20:
        raise ValueError('The number of frames is too low!')

    # Converting frames to grayscale
    if len(pixel_array.shape) == 4:
        gray_frames = pixel_array[:, :, :, 0]
    else:
        gray_frames = pixel_array

    # Cropping frames based on the information in the DICOM tags
    ReferencePixel = None
    if hasattr(dicom_dataset, 'SequenceOfUltrasoundRegions'):
        if dicom_dataset.SequenceOfUltrasoundRegions:
            if hasattr(dicom_dataset.SequenceOfUltrasoundRegions[0], 'RegionLocationMinY0'):
                Y0 = dicom_dataset.SequenceOfUltrasoundRegions[0].RegionLocationMinY0
            else:
                Y0 = 0
            if hasattr(dicom_dataset.SequenceOfUltrasoundRegions[0], 'RegionLocationMaxY1'):
                Y1 = dicom_dataset.SequenceOfUltrasoundRegions[0].RegionLocationMaxY1
            else:
                Y1 = gray_frames.shape[1]
            if hasattr(dicom_dataset.SequenceOfUltrasoundRegions[0], 'RegionLocationMinX0'):
                X0 = dicom_dataset.SequenceOfUltrasoundRegions[0].RegionLocationMinX0
            else:
                X0 = 0
            if hasattr(dicom_dataset.SequenceOfUltrasoundRegions[0], 'RegionLocationMaxX1'):
                X1 = dicom_dataset.SequenceOfUltrasoundRegions[0].RegionLocationMaxX1
            else:
                X1 = gray_frames.shape[2]
            gray_frames = gray_frames[:, Y0:Y1, X0:X1]

            if hasattr(dicom_dataset.SequenceOfUltrasoundRegions[0], 'ReferencePixelX0'):
                ReferencePixel = [dicom_dataset.SequenceOfUltrasoundRegions[0].ReferencePixelX0,
                                  dicom_dataset.SequenceOfUltrasoundRegions[0].ReferencePixelY0]
                if ReferencePixel[1] < 0:
                    ReferencePixel[1] = 0

    # Flipping frames (if necessary)
    if flip:
        gray_frames = np.flip(gray_frames, axis=2)

    # Checking whether the mask should be calculated based on a predefined set of frames
    if frame_limit is not None:
        first_n_gray_frames = gray_frames[:frame_limit]
    else:
        first_n_gray_frames = gray_frames

    # Creating and saving mask
    cropped_binary_mask, cropped_largest_contour_mask, cropped_hull_mask, bbox = \
        create_mask(first_n_gray_frames, ReferencePixel)
    os.makedirs(os.path.join(path_to_save, 'frames'), exist_ok=True)
    cv2.imwrite(os.path.join(path_to_save, 'mask.png'), cropped_hull_mask * 255)

    # Computing the extent of changes in pixel intensity values
    changes = np.zeros((first_n_gray_frames.shape[1], first_n_gray_frames.shape[2]))
    binary_mask = np.zeros((first_n_gray_frames.shape[1], first_n_gray_frames.shape[2]))
    for i in range(len(first_n_gray_frames) - 1):
        diff = abs(first_n_gray_frames[i].astype(np.int32) - first_n_gray_frames[i + 1].astype(np.int32))
        changes += diff

    nonzero_values = np.nonzero(changes)
    binary_mask[nonzero_values[0], nonzero_values[1]] += 1

    # Removing static objects
    filtered_gray_frames = np.zeros(gray_frames.shape)
    for i, gray_frame in enumerate(gray_frames):
        filtered_gray_frames[i, :, :] = np.where(binary_mask, gray_frame, 0)

    # Cropping, masking, and saving frames
    for i, filtered_gray_frame in enumerate(filtered_gray_frames):
        # Cropping
        cropped_frame = \
            filtered_gray_frame[int(bbox['min_x']):int(bbox['max_x']), int(bbox['min_y']):int(bbox['max_y'])]

        # Masking
        cropped_frame = np.where(cropped_hull_mask, cropped_frame, 0)

        # Saving
        im = Image.fromarray(cropped_frame)
        im = im.convert('L')
        im.save(os.path.join(path_to_save, 'frames', f'frame{str(i + 1)}.png'))


def preprocess_dicom(dicom_file_path: Path, output_path: Path, skip_saving: bool = False, flip: bool = False):
    # Creating a dictionary for the information extracted from DICOm tags
    data_dict = {}
    data_dict['dicom_path'] = str(dicom_file_path)

    # Loading data from the DICOM file
    dicom_dataset = pydicom.dcmread(dicom_file_path, stop_before_pixels=False)
    path_to_save = output_path / dicom_file_path.stem

    # Checking whether the DICOM file contains a 3D ultrasound video
    if is_3d_dicom(dicom_dataset):
        print(f'Error while processing {dicom_file_path}: 3D ultrasound videos are not supported.')
        return {}

    # Ensuring that the video has only one ultrasound region
    if hasattr(dicom_dataset, 'SequenceOfUltrasoundRegions') and len(dicom_dataset.SequenceOfUltrasoundRegions) > 1:
        print(f'Error while processing {dicom_file_path}: there are >1 ultrasound regions.')
        return {}

    if not hasattr(dicom_dataset, 'SequenceOfUltrasoundRegions') and not 'TOSHIBA' in dicom_dataset.Manufacturer:
        #TODO delete this
        if os.path.exists(path_to_save):
            os.system(f'rm -rf {path_to_save}')
        print(f'Error while processing {dicom_file_path}: there are no ultrasound regions.')
        return {}

    # Extracting frame rate from DICOM tags
    if 'RecommendedDisplayFrameRate' in dicom_dataset:
        data_dict['fps'] = int(dicom_dataset.RecommendedDisplayFrameRate)
    elif 'CineRate' in dicom_dataset:
        data_dict['fps'] = int(dicom_dataset.CineRate)
    elif 'FrameTime' in dicom_dataset:
        data_dict['fps'] = int(1000 / dicom_dataset.FrameTime)
    else:
        data_dict['fps'] = np.nan

    # Extracting heart rate from DICOM tags
    if 'HeartRate' in dicom_dataset:
        data_dict['HeartRate'] = int(dicom_dataset.HeartRate)
    else:
        data_dict['HeartRate'] = np.nan

    # Preprocessing and saving preprocessed frames
    dicom_id = dicom_file_path.stem
    data_dict['dicom_id'] = dicom_id
    if exists(path_to_save):
        print(f'{path_to_save} was already preprocessed.')
        return data_dict
    if not skip_saving:
        try:
            save_frames(dicom_dataset, path_to_save, flip=flip)
        except Exception as e:
            print(f'Error while processing {dicom_file_path}: {str(e)}')
            return {}

    print(f'{path_to_save} was preprocessed successfully.')
    return data_dict


def preprocess_multiprocess(input_csv: pd.DataFrame, output_folder: Path, output_csv_path: Path, skip_saving: bool):
    os.makedirs(output_folder, exist_ok=True)
    dicom_files_list = input_csv['dicom_path'].to_list()
    with multiprocessing.Pool(6) as pool:
        result = pool.starmap(preprocess_dicom,
                              [(Path(dicom_file_path), output_folder, skip_saving) for dicom_file_path in dicom_files_list])
    df = pd.DataFrame(result)
    df = pd.merge(input_csv, df, left_on='dicom_path', right_on='dicom_path', how='right')
    df.to_csv(output_csv_path, index=False)


def find_dicom_files(folder_path):
    dicom_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in [f for f in filenames]:
            dicom_files.append(os.path.join(dirpath, filename))
    return dicom_files


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('Input data',
                                      'Provide the path to the folder containing all DICOM files or'
                                      'a CSV file containing the paths to the DICOM files to be analyzed.')
    input_group = group.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--path_to_data', help='The path to the folder containing the DICOM files.')
    input_group.add_argument('--path_to_csv', help='The path to the CSV containing the paths to the DICOM files.')
    parser.add_argument('--output_folder', required=True,
                        help='The path to the folder where the preprocessed DICOM files will be saved.')
    parser.add_argument('--out_csv', default='codebook.csv', help='The name of the output CSV file.')
    parser.add_argument('--skip_saving', action='store_true',
                        help='Set whether the preprocessing and saving steps should be skipped.')

    # Finding the list of DICOM files to be preprocessed
    args = parser.parse_args()
    if args.path_to_data:
        dicom_files_list = find_dicom_files(args.path_to_data)
        df = pd.DataFrame(dicom_files_list, columns=['dicom_path'])
    else:
        df = pd.read_csv(args.path_to_csv)
        dicom_files_list = df.iloc[:, 0].to_list()

    output_folder = Path(args.output_folder)
    out_csv = Path(args.out_csv)
    skip_saving = args.skip_saving
    preprocess_multiprocess(input_csv=df, output_folder=output_folder,
                            output_csv_path=out_csv, skip_saving=skip_saving)


if __name__ == '__main__':
    main()
