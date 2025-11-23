from __future__ import annotations

from typing import Iterable

import torch

import config.configTrain as cfg
from infer_test import init_simulator


@torch.no_grad()
def generate_rollout(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    init_frame: torch.Tensor,
    actions: torch.Tensor,
    sample_step: int,
) -> torch.Tensor:
    """Run the diffusion model for a sequence of actions starting from init_frame."""

    if actions.numel() == 0:
        return torch.empty(0, *init_frame.shape, device=model.device)

    batch = {"observations": init_frame.unsqueeze(0).to(model.device)}
    zeta, _ = init_simulator(model, vae, batch)
    preds = []
    for action in actions:
        action_tensor = torch.tensor([int(action.item())], device=model.device).long()
        zeta, latent_obs = model.df_model.step(zeta, action_tensor.float(), sample_step)
        frame = vae.decode(latent_obs / cfg.scale_factor)
        frame = torch.clamp(frame, -1, 1)
        preds.append(frame.squeeze(0))
    return torch.stack(preds)
