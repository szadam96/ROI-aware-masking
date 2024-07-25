from transformers import VideoMAEForPreTraining
from transformers.models.videomae.modeling_videomae import VideoMAEForPreTrainingOutput, VideoMAEPreTrainedModel
from transformers.models.videomae.modeling_videomae import VideoMAEModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional

class CustomVideoMAEForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = VideoMAEForPreTraining(config)

    def forward(self, x, bool_masked_pos, video_mask=None):
        outputs = self.model(x, bool_masked_pos)

        if video_mask is None:
            return outputs
        
        logits = outputs.logits
        loss = outputs.loss


        frames = x

        batch_size, time, num_channels, height, width = frames.shape
        tubelet_size, patch_size = self.config.tubelet_size, self.config.patch_size

        patched_video_mask = F.max_pool2d(video_mask.unsqueeze(0).float(), self.config.patch_size).squeeze().bool()
        patched_video_mask = torch.stack([patched_video_mask for _ in range(self.config.num_frames//self.config.tubelet_size)], dim=1).flatten(start_dim=1)
        patched_video_mask = patched_video_mask[bool_masked_pos].reshape(batch_size, -1)
        with torch.no_grad():

            frames = frames.view(
                batch_size,
                time // tubelet_size,
                tubelet_size,
                num_channels,
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
            )
            # step 2: move dimensions to concatenate: (batch_size, T//ts, H//ps, W//ps, ts, ps, ps, C)
            frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
            # step 3: concatenate
            videos_patch = frames.view(
                batch_size,
                time // tubelet_size * height // patch_size * width // patch_size,
                tubelet_size * patch_size * patch_size * num_channels,
            )

        batch_size, _, num_channels = videos_patch.shape
        labels = videos_patch[bool_masked_pos].reshape(batch_size, -1, num_channels)

        loss_fn = nn.MSELoss()
        loss = loss_fn(logits[patched_video_mask], labels[patched_video_mask])

        #with torch.no_grad():
        #    logits[~patched_video_mask] = 0

        return VideoMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class CustomVideoMAEForRegression(VideoMAEPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.videomae = VideoMAEModel(config)

        # Classifier head
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        else:
            dropout = 0.2
        self.lvef_head = RegressorHead(config.hidden_size, 1, dropout=dropout)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        outputs = self.videomae(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        sequence_output = outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        lvef_pred = self.lvef_head(sequence_output)

        return {
            "lv_ef": lvef_pred,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }


class RegressorHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(self.dropout(x))))
        return self.fc2(x)