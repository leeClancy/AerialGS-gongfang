from __future__ import annotations

"""纯 PyTorch SSIM，无需 fused-ssim / 本机编译器。"""

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if pred.shape[1] not in (1, 3) and pred.shape[-1] in (1, 3):
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, device=pred.device, dtype=pred.dtype)
    kernel = kernel.expand(channels, 1, window_size, window_size).contiguous()
    padding = window_size // 2
    mu_x = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean()
