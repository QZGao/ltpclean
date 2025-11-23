from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader, TensorDataset

_DETECTOR_CACHE: dict[tuple[Path, torch.device], torch.nn.Module] = {}


def _get_detector(detector_path: Path, device: torch.device) -> torch.nn.Module:
    key = (detector_path.resolve(), device)
    cached = _DETECTOR_CACHE.get(key)
    if cached is not None:
        return cached

    if not detector_path.exists():
        raise FileNotFoundError(
            f"I3D TorchScript weights not found at {detector_path}. Please place the file manually."
        )

    detector = torch.jit.load(detector_path).eval().to(device)
    _DETECTOR_CACHE[key] = detector
    return detector


def _chunk_rgb(batch: torch.Tensor) -> torch.Tensor:
    channels = batch.size(1)
    if channels % 3 != 0:
        pad_channels = 3 - (channels % 3)
        pad = torch.zeros(
            batch.size(0),
            pad_channels,
            batch.size(2),
            batch.size(3),
            batch.size(4),
            device=batch.device,
        )
        batch = torch.cat([batch, pad], dim=1)
    return torch.cat(torch.chunk(batch, chunks=batch.size(1) // 3, dim=1), dim=0)


def _compute_stats(
    videos: torch.Tensor,
    detector: torch.nn.Module,
    batch_size: int,
    max_items: Optional[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if max_items is not None:
        videos = videos[: max_items]
    dataset = TensorDataset(videos)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    for (batch,) in loader:
        batch = batch.to(device, dtype=torch.float32)
        batch = _chunk_rgb(batch)
        outputs = detector(batch, rescale=True, resize=True, return_features=True)
        features.append(outputs.detach().cpu().numpy())
    stacked = np.concatenate(features, axis=0)
    mu = np.mean(stacked, axis=0)
    sigma = np.cov(stacked, rowvar=False)
    return mu, sigma


def compute_fvd(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    max_items: Optional[int] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    detector_path: Optional[Path] = None,
) -> float:
    """Compute Frechet Video Distance between two video sets.

    Args:
        y_true: Tensor shaped [N, C, T, H, W] for reference videos in [0, 1].
        y_pred: Tensor shaped [N, C, T, H, W] for generated videos in [0, 1].
        max_items: Optional cap on number of videos sampled from each set.
        batch_size: Mini-batch size for the feature extractor.
        device: Torch device; defaults to CUDA if available.
        detector_path: Location of the TorchScript I3D weights. Defaults to
            metrics/i3d_torchscript.pt next to this package.
    """

    if y_true.ndim != 5 or y_pred.ndim != 5:
        raise ValueError("FVD expects inputs shaped [N, C, T, H, W]")
    if y_true.shape[1:] != y_pred.shape[1:]:
        raise ValueError("Reference and predicted videos must share shape")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_path = detector_path or Path(__file__).resolve().with_name("i3d_torchscript.pt")

    detector = _get_detector(detector_path, device)
    mu_true, sigma_true = _compute_stats(y_true, detector, batch_size, max_items, device)
    mu_pred, sigma_pred = _compute_stats(y_pred, detector, batch_size, max_items, device)

    diff = mu_pred - mu_true
    covmean, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma_pred.shape[0]) * eps
        covmean = scipy.linalg.sqrtm(np.dot(sigma_pred + offset, sigma_true + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = float(diff.dot(diff) + np.trace(sigma_pred + sigma_true - 2 * covmean))
    if math.isnan(fvd):
        raise ValueError("FVD computation returned NaN; check inputs")
    return fvd
