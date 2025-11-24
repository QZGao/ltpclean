from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import json

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

try:  # TorchMetrics currently lacks FVD on most releases.
    from torchmetrics.image.fvd import FrechetVideoDistance

    _HAS_TORCHMETRICS_FVD = True
except ImportError:  # pragma: no cover - fallback path
    from .third_party.fvd_metric import compute_fvd as _compute_fvd

    _HAS_TORCHMETRICS_FVD = False

from .base import BaseEvaluator, EvaluationContext
from .data import TrajectoryDataset, get_dataset
from .rollout import generate_rollout


def _to_zero_one(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-1.0, 1.0).add(1.0).mul(0.5)


def _batch_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute PSNR assuming inputs are in [0, 1] range (converts to 255 scale)."""
    # Scale [0,1] to [0,255] for standard PSNR calculation
    pred_255 = pred * 255.0
    target_255 = target * 255.0
    mse = torch.mean((pred_255 - target_255) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(255.0 ** 2 / (mse + eps))
    return psnr


class VisualQualityEvaluator(BaseEvaluator):
    name = "visual"
    description = "Perceptual metrics such as LPIPS/PSNR/FID/FVD"

    def evaluate(self, context: EvaluationContext) -> Dict[str, Any]:
        progress_file = context.output_dir / f"{self.name}.progress.json"
        progress_data = _load_progress(progress_file)
        dataset = get_dataset(context)
        num_requested = max(1, context.extra_kwargs.get("num_trajectories", 600))
        seeds = context.seeds or [0]

        for horizon in context.prediction_lengths:
            key = f"len_{horizon}"
            available = dataset.available_windows(horizon)
            if available <= 0:
                progress_data[key] = {
                    "status": "skipped",
                    "reason": "insufficient_data",
                }
                _save_progress(progress_file, progress_data)
                continue

            sample_total = min(num_requested, available)
            per_seed = max(1, math.ceil(sample_total / len(seeds)))
            horizon_entry = progress_data.get(key, {})
            horizon_results = horizon_entry.get("results", {})

            for step in context.sample_steps:
                step_key = f"steps_{step}"
                if step_key in horizon_results:
                    continue
                metrics = self._evaluate_combo(
                    context=context,
                    dataset=dataset,
                    horizon=horizon,
                    sample_step=step,
                    per_seed=per_seed,
                    sample_total=sample_total,
                    seeds=seeds,
                )
                horizon_results[step_key] = metrics
                horizon_entry["status"] = "ok"
                horizon_entry["results"] = horizon_results
                progress_data[key] = horizon_entry
                _save_progress(progress_file, progress_data)

            progress_data[key] = {
                "status": "ok",
                "results": horizon_results,
            }
            _save_progress(progress_file, progress_data)

        final_data = {
            "status": "ok",
            "details": progress_data,
        }
        if progress_file.exists():
            progress_file.unlink(missing_ok=True)
        return final_data

    def _evaluate_combo(
        self,
        context: EvaluationContext,
        dataset: TrajectoryDataset,
        horizon: int,
        sample_step: int,
        per_seed: int,
        sample_total: int,
        seeds: List[int],
    ) -> Dict[str, Any]:
        device = context.device
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        fvd_metric = None
        video_preds: List[torch.Tensor] | None = None
        video_targets: List[torch.Tensor] | None = None
        if horizon > 1:
            if _HAS_TORCHMETRICS_FVD:
                fvd_metric = FrechetVideoDistance(feature=2048).to(device)
            else:
                video_preds = []
                video_targets = []

        total_lpips = 0.0
        total_psnr = 0.0
        frame_counter = 0
        video_counter = 0
        
        # Per-frame metrics for specific timesteps
        per_frame_metrics_enabled = context.extra_kwargs.get("per_frame_metrics", False)
        per_frame_stats = {}  # {timestep: {lpips: [], psnr: []}}

        processed = 0
        with torch.inference_mode():
            for seed in seeds:
                remaining = sample_total - processed
                if remaining <= 0:
                    break
                windows = dataset.sample_windows(horizon, min(per_seed, remaining), seed)
                processed += len(windows)
                for window in windows:
                    preds = generate_rollout(
                        model=context.model,
                        vae=context.vae,
                        init_frame=window.init_frame,
                        actions=window.actions,
                        sample_step=sample_step,
                    )
                    targets = window.target_frames.to(device)
                    if preds.shape[0] != targets.shape[0]:
                        continue
                    preds = preds.to(device)
                    preds_norm = preds.clamp(-1, 1)
                    targets_norm = targets.clamp(-1, 1)
                    preds_01 = _to_zero_one(preds_norm)
                    targets_01 = _to_zero_one(targets_norm)

                    fid_metric.update(targets_01, real=True)
                    fid_metric.update(preds_01, real=False)

                    if fvd_metric is not None:
                        fvd_metric.update(targets_01.unsqueeze(0), real=True)
                        fvd_metric.update(preds_01.unsqueeze(0), real=False)
                    elif video_preds is not None and video_targets is not None:
                        video_targets.append(targets_01.permute(1, 0, 2, 3).unsqueeze(0).cpu())
                        video_preds.append(preds_01.permute(1, 0, 2, 3).unsqueeze(0).cpu())

                    lpips_val = lpips_metric(preds_norm, targets_norm)
                    total_lpips += lpips_val.item() * preds_norm.shape[0]
                    psnr_vals = _batch_psnr(preds_01, targets_01)
                    total_psnr += psnr_vals.sum().item()
                    
                    # Collect per-frame metrics for specific timesteps
                    if per_frame_metrics_enabled:
                        timesteps_to_track = [4, 8, 16, 32, 64]
                        for t_idx in range(preds_norm.shape[0]):
                            timestep = t_idx + 1  # 1-indexed
                            if timestep in timesteps_to_track:
                                if timestep not in per_frame_stats:
                                    per_frame_stats[timestep] = {"lpips": [], "psnr": []}
                                
                                # Compute metrics for this specific frame
                                frame_lpips = lpips_metric(
                                    preds_norm[t_idx:t_idx+1], 
                                    targets_norm[t_idx:t_idx+1]
                                ).item()
                                frame_psnr = psnr_vals[t_idx].item()
                                
                                per_frame_stats[timestep]["lpips"].append(frame_lpips)
                                per_frame_stats[timestep]["psnr"].append(frame_psnr)
                    
                    frame_counter += preds_norm.shape[0]
                    video_counter += 1

        if frame_counter == 0:
            return {
                "status": "skipped",
                "reason": "no_frames",
            }

        fid_score = fid_metric.compute().item()
        result = {
            "status": "ok",
            "num_frames": frame_counter,
            "num_sequences": video_counter,
            "lpips": total_lpips / frame_counter,
            "psnr": total_psnr / frame_counter,
            "fid": fid_score,
        }
        
        # Add per-frame metrics if enabled
        if per_frame_metrics_enabled and per_frame_stats:
            per_frame_summary = {}
            for timestep, stats in sorted(per_frame_stats.items()):
                if stats["lpips"]:
                    per_frame_summary[f"frame_{timestep}"] = {
                        "count": len(stats["lpips"]),
                        "lpips": sum(stats["lpips"]) / len(stats["lpips"]),
                        "psnr": sum(stats["psnr"]) / len(stats["psnr"]),
                    }
            if per_frame_summary:
                result["per_frame"] = per_frame_summary

        if fvd_metric is not None and video_counter > 0:
            result["fvd"] = fvd_metric.compute().item()
        elif video_preds:
            preds_tensor = torch.cat(video_preds, dim=0)
            targets_tensor = torch.cat(video_targets, dim=0)
            # I3D network requires minimum 16 frames after downsampling
            # Skip FVD for short horizons (typically len_1 and len_16 are too short)
            if preds_tensor.shape[2] >= 16:
                result["fvd"] = _compute_fvd(
                    targets_tensor,
                    preds_tensor,
                    device=device,
                    batch_size=4,
                )
            else:
                result["fvd"] = None
        else:
            result["fvd"] = None

        return result


def _load_progress(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_progress(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
