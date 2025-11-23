from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class VideoActionModel(nn.Module):
    """A lightweight 3D CNN used to score action-consistency."""

    def __init__(self, num_actions: int, window: int = 17):
        super().__init__()
        self.window = window
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = self.pool(feats).flatten(1)
        return self.classifier(pooled)


_CACHE_KEY = "vam_model"


def _load_state_dict(model: nn.Module, state_dict):
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
    model_params = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if name not in model_params:
            continue
        if model_params[name].shape != value.shape:
            print(
                f"Skipping VAM state {name} (checkpoint shape {value.shape}, model expects {model_params[name].shape})"
            )
            continue
        filtered[name] = value
    model.load_state_dict(filtered, strict=False)


def get_vam_model(
    context,
    num_actions: int,
) -> Optional[VideoActionModel]:
    cache = context.extra_kwargs.setdefault("cache", {})
    ckpt_path = context.extra_kwargs.get("vam_checkpoint")
    if not ckpt_path:
        return None
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None
    window = int(context.extra_kwargs.get("vam_window", 17))
    cached = cache.get(_CACHE_KEY)
    if cached:
        cached_ckpt, cached_actions, cached_window, model = cached
        if cached_ckpt == ckpt_path and cached_actions == num_actions and cached_window == window:
            return model
    model = VideoActionModel(num_actions=num_actions, window=window)
    state = torch.load(ckpt_path, map_location=context.device)
    _load_state_dict(model, state)
    model.to(context.device)
    model.eval()
    cache[_CACHE_KEY] = (ckpt_path, num_actions, window, model)
    return model
