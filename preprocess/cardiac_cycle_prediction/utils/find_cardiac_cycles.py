import os
import numpy as np
import torch
from scipy.signal import find_peaks
from pathlib import Path

def extract_idx_from_name(name):
    stem = Path(name).stem
    #print(stem)
    assert stem[:5] == 'frame'
    idx_str = stem[5:]
    return int(idx_str)

def es_prediction(input_folder, fps, max_hr=150):
    frames_for_max_hr = (60 / max_hr) * fps

    pred_path = input_folder / 'es_preds'
    prediction_filenames = os.listdir(pred_path)
    prediction_filenames.sort(key=extract_idx_from_name)

    predictions = [np.load(os.path.join(pred_path, filename)) for filename in prediction_filenames]
    predictions = np.stack(predictions)
    predictions = torch.from_numpy(predictions)
    probabilities = predictions.tanh()
    min_probability = np.abs(np.min(probabilities.numpy()))
    min_max_proba_diff = np.abs(np.max(probabilities.numpy()) - np.min(probabilities.numpy()))

    es, _ = find_peaks(-probabilities,
                       height=min_probability * 0.1,
                       prominence=min_max_proba_diff * 0.15,
                       distance=frames_for_max_hr)

    es_times = [list(range(es[i], es[i + 1])) for i in range(len(es) - 1)]

    return es_times