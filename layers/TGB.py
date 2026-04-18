# layers/TGB.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import Inception_Block_V1



class TGB(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.use_tgm = getattr(configs, 'TGB', 1)  # 默认开启
        self.tgb_mode = getattr(configs, 'TGB_mode', 'all')
        self.batch = configs.batch_size

        num_kernels = configs.num_kernels
        self.context_window = configs.seq_len
        self.target_window = configs.pred_len

        # Temporal Gradient Convolution Path
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.enc_in, configs.d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.enc_in, num_kernels=num_kernels)
        )

        self.task_adaptive = nn.Parameter(torch.zeros(1, 2))

        if self.tgb_mode == 'all':
            self.weights = nn.Parameter(torch.randn(self.batch, 2))
        elif self.tgb_mode in ['Acc', 'Trend']:
            self.weights = nn.Parameter(torch.randn(self.batch, 1))
        else:
            raise ValueError(f"Unsupported TGB_mode: {self.tgb_mode}")

        self.feature_to_pred = nn.Conv1d(in_channels=self.context_window, out_channels=self.target_window, kernel_size=1)

    def forward(self, x):  # [B, T, N]
        if not self.use_tgm:
            return torch.zeros_like(x)

        B, T, N = x.size()
        grad = compute_temporal_gradient_tensor(x).permute(0, 1, 2, 3)  # [B, N, T, 2]
        feat = self.conv(grad).permute(0, 2, 1, 3)  # [B, T, N, 2]

        C = feat.size(-1)
        weights = self.weights[:, :C].unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        gating = torch.softmax(self.task_adaptive, dim=-1)

        feat = feat * weights * gating.view(1, 1, 1, 2)
        out = torch.sum(feat, dim=-1).permute(0, 2, 1)  # [B, N, T]
        out = self.feature_to_pred(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out


def compute_temporal_gradient_tensor(x: torch.Tensor) -> torch.Tensor:

    """
    [Times2D](https://arxiv.org/abs/2504.00118).
    Compute trend and acceleration signals (1st and 2nd derivatives) from the input sequence.

    Args:
        x: Input tensor of shape [B, T, N], where B is batch size,
           T is sequence length, and N is number of variables.

    Returns:
        A tensor of shape [B, N, T, 2], where the last dimension represents:
            [:, :, :, 0] - Trend signal (1st derivative)
            [:, :, :, 1] - Acceleration signal (2nd derivative)
    """

    trend_signal = x[:, 1:] - x[:, :-1]
    trend_signal = torch.cat([torch.zeros_like(trend_signal[:, :1, :]), trend_signal], dim=1)

    acc_signal = trend_signal[:, 1:] - trend_signal[:, :-1]
    acc_signal = torch.cat([torch.zeros_like(acc_signal[:, :1, :]), acc_signal], dim=1)

    temporal_gradient_tensor = torch.stack([trend_signal, acc_signal], dim=-1)
    temporal_gradient_tensor = temporal_gradient_tensor.permute(0, 2, 1, 3)
    return temporal_gradient_tensor





