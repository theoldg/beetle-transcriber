import torch
from torch import Tensor

from beetle_transcriber.midi import Channel


def decode_naive(model_output: Tensor, nms_radius: int = 2, threshold: float = .7):
    batch, n_time, n_freq, channels = model_output.shape
    c_max = torch.sigmoid(model_output[..., Channel.CONFIDENCE_MAX])
    window = 2 * nms_radius + 1
    # reshape -> (batch * n_freq, 1, n_time) so pool runs along time.
    c_for_pool = c_max.permute(0, 2, 1).reshape(batch * n_freq, 1, n_time)
    c_pooled = torch.nn.functional.max_pool1d(
        c_for_pool,
        kernel_size=window,
        stride=1,
        padding=nms_radius,
    )
    # Reshape back
    c_pooled = c_pooled.reshape(batch, n_freq, n_time).permute(0, 2, 1)
    is_local_max = (c_max == c_pooled)
    is_above_thresh = (c_max >= threshold)
    
    note_positions = is_local_max & is_above_thresh
    ...
    
