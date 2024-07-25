import os
from pathlib import Path
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter


class ValEveryNSteps(pl.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.run_evaluation(test_mode=False)


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write_on_batch_end(self, trainer, pl_module, pred_step_output, *args, **kwargs):
        outputs, dicom_ids, filenames = pred_step_output
        for output, dicom_id, filename in zip(outputs, dicom_ids, filenames):
            output = output.detach().cpu().numpy()
           
            output_folder = os.path.join(self.output_dir, dicom_id, 'es_preds')
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            output_filename = Path(filename).stem
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, output)
