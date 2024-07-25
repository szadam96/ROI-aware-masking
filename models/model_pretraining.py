import numpy as np
import torch
import torch.nn as nn
import lightning as pl
from models.architectures import CustomVideoMAEForPretraining
from utils.train_utils import get_optimizer, get_scheduler, get_pretraining_model, get_model_config
import torch.nn.functional as F

class EchoModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.lr = self.config['training']['learning_rate']
        self.mae_config = get_model_config(config['model'])
        self.model = get_pretraining_model(config)

        self.apply(self._init_weights)

        self.best_loss = 1000
        self.save_hyperparameters(config)
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def _init_weights(self, m):
        '''Initialize the weights of the model'''
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        '''Configure the optimizer and scheduler'''
        optimizer = get_optimizer(self.config['training'], self)
        scheduler = get_scheduler(self.config['training'], optimizer)
        return [optimizer], [scheduler]
    

    def training_step(self, batch, batch_idx):
        '''Training step for the model'''
        #torch.cuda.empty_cache()
        input_tensor = batch['input_tensor']
        bool_mask = batch['bool_masked_pos']
        if isinstance(self.model, CustomVideoMAEForPretraining):
            model_output = self.model(input_tensor, bool_mask, batch['binary_mask'])
        else:
            model_output = self.model(input_tensor, bool_mask)
        loss = model_output.loss

        output = {'loss': loss.item()}
        self.train_outputs.append(output)
        return loss

    def on_train_epoch_end(self):
        '''Training epoch end for the model'''
        outputs = self.train_outputs
        if self.trainer.is_global_zero and len(outputs) != 0:
            avg_loss = np.array([x['loss'] for x in outputs]).mean()
            self.log('train_loss', avg_loss, prog_bar=False, logger=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], logger=True)
        self.train_outputs = []

    
    def validation_step(self, batch, batch_idx):
        input_tensor = batch['input_tensor']
        bool_mask = batch['bool_masked_pos']
        if isinstance(self.model, CustomVideoMAEForPretraining):
            binary_mask = batch['binary_mask']
            model_output = self.model(input_tensor, bool_mask, binary_mask)
        else:
            model_output = self.model(input_tensor, bool_mask)
        loss = model_output.loss
        logits = model_output.logits

        output = {'loss': loss.detach()}
        self.val_outputs.append(output)
        if batch_idx == 0 and self.trainer.is_global_zero:
            self.plot_reconsturction(input_tensor.detach().cpu(), bool_mask.detach().cpu(), logits.detach().cpu(), binary_mask.detach().cpu())
        return loss
    
    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=False, logger=True, sync_dist=True)
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.trainer.save_checkpoint(str(self.logger.save_dir) + '/best_loss.ckpt')
        self.val_outputs = []

    def plot_reconsturction(self, input_tensor, bool_mask, logits, video_mask):
        input_tensor = input_tensor.float()
        logits = logits.float()
        video_mask = video_mask.float()
        batch_size = input_tensor.shape[0]
        patched_video_mask = F.max_pool2d(video_mask.unsqueeze(0).float(), self.mae_config.patch_size).squeeze().bool()
        patched_video_mask = torch.stack([patched_video_mask for _ in range(self.mae_config.num_frames//self.mae_config.tubelet_size)], dim=1).flatten(start_dim=1)
        patched_video_mask = patched_video_mask[bool_mask].reshape(batch_size, -1)

        logits[~patched_video_mask] = 0

        original_patches = input_tensor.view(
            batch_size, 
            self.mae_config.num_frames // self.mae_config.tubelet_size,
            self.mae_config.tubelet_size,
            1,
            self.mae_config.image_size // self.mae_config.patch_size,
            self.mae_config.patch_size,
            self.mae_config.image_size // self.mae_config.patch_size,
            self.mae_config.patch_size
        )
        original_patches = original_patches.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous().view(
            batch_size, 
            self.mae_config.num_frames // self.mae_config.tubelet_size * (self.mae_config.image_size // self.mae_config.patch_size) ** 2,
            self.mae_config.tubelet_size * self.mae_config.patch_size ** 2
        )
        reconstructed_patches = original_patches.clone()
        reconstructed_patches[bool_mask] = logits.reshape(logits.shape[0]*logits.shape[1], logits.shape[2])
        reconsturcted_frames = reconstructed_patches.view(
            batch_size, 
            self.mae_config.num_frames // self.mae_config.tubelet_size,
            self.mae_config.image_size // self.mae_config.patch_size,
            self.mae_config.image_size // self.mae_config.patch_size,
            self.mae_config.tubelet_size,
            self.mae_config.patch_size,
            self.mae_config.patch_size
        )
        reconsturcted_frames = reconsturcted_frames.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(
            batch_size, 
            self.mae_config.num_frames,
            self.mae_config.image_size,
            self.mae_config.image_size
        )
        #self.logger.experiment[f'predictions/original/{i}_es'].append(original_image_es)
        #self.logger.experiment[f'predictions/original/{i}_ed'].append(original_image_ed)
        self.logger.experiment.add_image(f'predictions/reconstructed/0_es', reconsturcted_frames[0, 0, :, :].numpy(), dataformats='HW', global_step=self.current_epoch)
        self.logger.experiment.add_image(f'predictions/reconstructed/0_ed', reconsturcted_frames[0, 7, :, :].numpy(), dataformats='HW', global_step=self.current_epoch)

