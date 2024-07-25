from . import architecture

import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics import F1Score, Accuracy

from preprocess.cardiac_cycle_prediction.utils.utils import get_name, get_class
from preprocess.cardiac_cycle_prediction.utils import create_from_config

class EchoLightningModule(pl.LightningModule):
    def __init__(self, model, criterion=None, optimizer=None, **kwargs):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.criterion_name = get_name(self.criterion)
        self.optimizer = optimizer

        num_classes = 2
        self.eval_criterions = nn.ModuleList([
            F1Score(task='multiclass', num_classes=num_classes),
            Accuracy(task='multiclass', num_classes=num_classes)
        ])

        #self.save_hyperparameters()

        self.train_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, batch, batch_idx):
        loss, output, target = self._common_step(batch, batch_idx)
        self.train_step_outputs.append((output.detach(), target.detach()))

        return loss
    
    def on_train_epoch_end(self):
        outputs = torch.concat([output for output, _ in self.train_step_outputs])
        targets = torch.concat([target for _, target in self.train_step_outputs])
        metric_scores = self._compute_metrics(outputs, targets, 'train')

        self.log_dict(metric_scores, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)    
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, output, target = self._common_step(batch, batch_idx)
        self.validation_step_outputs.append((output.detach(), target.detach()))

        return loss

    def on_validation_epoch_end(self):
        outputs = torch.concat([output for output, _ in self.validation_step_outputs])
        targets = torch.concat([target for _, target in self.validation_step_outputs])
        metric_scores = self._compute_metrics(outputs, targets, 'val')

        self.log_dict(metric_scores, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)    
    
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return self.optimizer


    def _compute_metrics(self, output, target, phase):
        metric_scores = {
            f'{phase}_{self.criterion_name}': self.criterion(output, target)
        }
        for eval_criterion in self.eval_criterions:
            name = get_name(eval_criterion)
            full_name = f'{phase}_{name}' 
            metric_scores[full_name] = eval_criterion(output, target)
        
        return metric_scores

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch['input'])
        return output, batch['dicom_id'], batch['filename']



class Echo2dPicturesNetLM(EchoLightningModule):
    def __init__(self, *args, **kwargs):
        super(Echo2dPicturesNetLM, self).__init__(*args, **kwargs)

    def _common_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']
        output = self.forward(input)
        loss = self.criterion(output, target)

        return loss, output, target
    

class Echo2dVideosNetLM(EchoLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, batch_size, n_frames):
        output, _ = self.model(input, batch_size, n_frames)
        return output

    def _common_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']
        
        # Batch x Frames x Channels x Height x Width
        # assert len(input.shape) == 5
        batch_size, n_frames, *img_shape = input.shape
        _, _, *label_shape = target.shape
        input = input.reshape(-1, *img_shape)
        target = target.reshape(-1, *label_shape)

        output = self.forward(input, batch_size, n_frames)
        _, _, *output_shape = output.shape
        output = output.reshape(-1, *output_shape)
        loss = self.criterion(output, target)
        
        return loss, output, target

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        batch_size, n_frames, *img_shape = input.shape
        input = input.reshape(-1, *img_shape)

        output = self(input, batch_size, n_frames)
        _, _, *output_shape = output.shape
        output = output.reshape(-1, *output_shape)

        dicom_ids = []
        filenames = []
        for batch_idx in range(batch_size):
            for frame_idx in range(n_frames):
                dicom_ids.append(batch['dicom_id'][frame_idx][batch_idx])
                filenames.append(batch['filename'][frame_idx][batch_idx])

        return output, dicom_ids, filenames

def create_model_module(module_config):
    module_class = get_class(module_config['name'], modules=['cardiac_cycle_prediction.model.model'])
    module_config.pop('name')
    
    model_config = module_config.pop('model')
    model = architecture.create_model(model_config)

    loss_config = module_config.pop('loss')
    criterion = create_from_config.create_loss(loss_config)

    optimizer_config = module_config.pop('optimizer')
    optimizer = create_from_config.create_optimizer(optimizer_config, model)


    return module_class(model=model, criterion=criterion, optimizer=optimizer, **module_config)
