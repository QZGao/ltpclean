from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .base import BaseEvaluator, EvaluationContext
from .data import TrajectoryDataset, get_dataset
from .rollout import generate_rollout
from .vam import get_vam_model


def _to_vam_input(frames: torch.Tensor) -> torch.Tensor:
    clips = frames.clamp(-1.0, 1.0).add(1.0).mul(0.5)
    return clips.unsqueeze(0).permute(0, 2, 1, 3, 4)


class MechanicsAccuracyEvaluator(BaseEvaluator):
    name = "mechanics"
    description = "Action-aware accuracy metrics (ActAcc/ProbDiff)"

    def evaluate(self, context: EvaluationContext) -> Dict[str, Any]:
        dataset = get_dataset(context)
        vam = get_vam_model(context, dataset.num_actions)
        if vam is None:
            return {
                "status": "skipped",
                "reason": "vam_checkpoint_missing",
            }

        window = max(2, int(context.extra_kwargs.get("vam_window", 17)))
        seeds = context.seeds or [0]
        num_requested = max(1, context.extra_kwargs.get("num_trajectories", 600))

        results: Dict[str, Any] = {}
        for horizon in context.prediction_lengths:
            key = f"len_{horizon}"
            if horizon < window:
                results[key] = {
                    "status": "skipped",
                    "reason": "horizon_lt_vam_window",
                }
                continue

            available = dataset.available_windows(horizon)
            if available <= 0:
                results[key] = {
                    "status": "skipped",
                    "reason": "insufficient_data",
                }
                continue

            sample_total = min(num_requested, available)
            per_seed = max(1, math.ceil(sample_total / len(seeds)))
            horizon_results: Dict[str, Any] = {}

            for step in context.sample_steps:
                metrics = self._evaluate_combo(
                    context=context,
                    dataset=dataset,
                    vam=vam,
                    horizon=horizon,
                    sample_step=step,
                    per_seed=per_seed,
                    sample_total=sample_total,
                    seeds=seeds,
                    window=window,
                )
                horizon_results[f"steps_{step}"] = metrics

            results[key] = {
                "status": "ok",
                "results": horizon_results,
            }

        return {
            "status": "ok",
            "details": results,
        }

    def _evaluate_combo(
        self,
        context: EvaluationContext,
        dataset: TrajectoryDataset,
        vam: torch.nn.Module,
        horizon: int,
        sample_step: int,
        per_seed: int,
        sample_total: int,
        seeds: List[int],
        window: int,
    ) -> Dict[str, Any]:
        act_correct = 0
        act_total = 0
        prob_diff_sum = 0.0
        prob_count = 0
        processed = 0

        with torch.inference_mode():
            for seed in seeds:
                remaining = sample_total - processed
                if remaining <= 0:
                    break
                windows = dataset.sample_windows(horizon, min(per_seed, remaining), seed)
                processed += len(windows)
                for window_bundle in windows:
                    preds = generate_rollout(
                        model=context.model,
                        vae=context.vae,
                        init_frame=window_bundle.init_frame,
                        actions=window_bundle.actions,
                        sample_step=sample_step,
                    )
                    if preds.shape[0] != window_bundle.target_frames.shape[0]:
                        continue
                    sequence_pred = torch.cat(
                        [window_bundle.init_frame.unsqueeze(0).to(context.device), preds.to(context.device)], dim=0
                    )
                    sequence_gt = torch.cat(
                        [window_bundle.init_frame.unsqueeze(0).to(context.device), window_bundle.target_frames.to(context.device)],
                        dim=0,
                    )
                    actions = window_bundle.actions
                    max_start = sequence_pred.shape[0] - window
                    if max_start < 1:
                        continue
                    for start in range(0, max_start + 1):
                        action_idx = start + window - 2
                        if action_idx < 0 or action_idx >= actions.shape[0]:
                            continue
                        label = int(actions[action_idx].item())
                        pred_clip = sequence_pred[start : start + window]
                        gt_clip = sequence_gt[start : start + window]
                        pred_probs = torch.softmax(vam(_to_vam_input(pred_clip)), dim=-1)
                        gt_probs = torch.softmax(vam(_to_vam_input(gt_clip)), dim=-1)
                        act_correct += int(pred_probs.argmax(dim=-1).item() == label)
                        act_total += 1
                        prob_diff = F.l1_loss(pred_probs, gt_probs, reduction="mean")
                        prob_diff_sum += prob_diff.item()
                        prob_count += 1

        if act_total == 0 or prob_count == 0:
            return {
                "status": "skipped",
                "reason": "no_valid_windows",
            }

        return {
            "status": "ok",
            "act_acc": act_correct / act_total,
            "prob_diff": prob_diff_sum / prob_count,
            "evaluated_transitions": act_total,
        }
