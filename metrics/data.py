from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from config import configTrain as cfg
from utils import read_file

Tensor = torch.Tensor
MIN_VAM_ACTIONS = 7


@dataclass
class TrajectoryWindow:
    """Represents a contiguous rollout window extracted from the dataset."""

    init_frame: Tensor  # [3, H, W] in [-1, 1]
    target_frames: Tensor  # [T, 3, H, W] in [-1, 1]
    actions: Tensor  # [T]


class TrajectoryDataset:
    """Utility for sampling rollout windows from flattened gameplay logs."""

    def __init__(self, data_path: Path, device: torch.device, data_type: str = cfg.data_type):
        self.data_path = data_path
        self.device = device
        self.data_type = data_type
        self._frames: Tensor | None = None
        self._actions: Tensor | None = None
        self._load()

    def _load(self) -> None:
        raw = read_file(str(self.data_path), self.data_type)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError(f"Invalid dataset shape {raw.shape} from {self.data_path}")
        pixel_count = raw.shape[1] - 1
        expected_pixels = cfg.img_size * cfg.img_size * 3
        if pixel_count < expected_pixels:
            raise ValueError(
                f"Dataset images have {pixel_count} values but expected at least {expected_pixels}."
            )
        frame_data = raw[:, :expected_pixels]
        action_data = raw[:, -1]
        frames = torch.tensor(frame_data, dtype=torch.float32)
        frames = frames.view(-1, cfg.img_size, cfg.img_size, 3)
        frames = frames.permute(0, 3, 1, 2)  # [N, 3, H, W]
        frames = frames / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
        actions = torch.tensor(action_data, dtype=torch.long)
        self._frames = frames
        self._actions = actions

    @property
    def num_frames(self) -> int:
        return self._frames.shape[0]

    @property
    def frames(self) -> Tensor:
        if self._frames is None:
            raise RuntimeError("Trajectory data has not been loaded yet")
        return self._frames

    @property
    def actions(self) -> Tensor:
        if self._actions is None:
            raise RuntimeError("Trajectory data has not been loaded yet")
        return self._actions

    def available_windows(self, horizon: int) -> int:
        if horizon <= 0:
            return 0
        return max(0, self.num_frames - (horizon + 1))

    @property
    def num_actions(self) -> int:
        return int(max(self._actions.max().item() + 1, MIN_VAM_ACTIONS))

    def sample_windows(
        self,
        horizon: int,
        count: int,
        seed: int,
        stride: int = 1,
    ) -> List[TrajectoryWindow]:
        max_start = self.available_windows(horizon)
        if max_start <= 0:
            return []
        indices = list(range(0, max_start, stride))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = indices[: min(count, len(indices))]
        windows: List[TrajectoryWindow] = []
        for idx in selected:
            frames = self._frames[idx : idx + horizon + 1]
            init_frame = frames[0].to(self.device)
            target_frames = frames[1:].to(self.device)
            actions = self._actions[idx : idx + horizon].to(self.device)
            windows.append(
                TrajectoryWindow(
                    init_frame=init_frame,
                    target_frames=target_frames,
                    actions=actions,
                )
            )
        return windows

    def clip_indices(self, window: int, stride: int = 1) -> List[int]:
        if window < 2:
            raise ValueError("VAM clips require at least two frames per window")
        max_start = min(self.num_frames - window, self._actions.shape[0] - (window - 1))
        if max_start <= 0:
            return []
        return list(range(0, max_start, stride))


_DATASET_CACHE_KEY = "trajectory_dataset"


def get_dataset(context) -> TrajectoryDataset:
    cache = context.extra_kwargs.setdefault("cache", {})
    dataset = cache.get(_DATASET_CACHE_KEY)
    if dataset is None:
        dataset = TrajectoryDataset(context.dataset_path, context.device)
        cache[_DATASET_CACHE_KEY] = dataset
    return dataset
