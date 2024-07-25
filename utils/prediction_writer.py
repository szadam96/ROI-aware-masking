import os
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import lightning as pl
from lightning.pytorch.callbacks import BasePredictionWriter

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, **kwargs):
        super().__init__('epoch', **kwargs)
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, predictions, batch_indices) -> None:
        print(predictions)
        lvef_preds = torch.cat([x['lvef_pred'] for x in predictions])
        lvef_trues = torch.cat([x['lvef_true'] for x in predictions])

        validation_df = pd.DataFrame({'patient_id': np.concatenate([x['patient_id'] for x in predictions]),
                                        'dicom_id': np.concatenate([x['dicom_id'] for x in predictions]),
                                        'lvef_true': [x for x in lvef_trues.numpy()],
                                        'lvef_pred': [x for x in lvef_preds.numpy()]})
        
        validation_df.to_csv(self.output_dir / f'validation_predictions.csv', index=False)