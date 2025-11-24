from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import torch

from .base import BaseEvaluator, EvaluationContext
from .data import TrajectoryDataset, get_dataset
from .rollout import generate_rollout


class InteractionEfficiencyEvaluator(BaseEvaluator):
    name = "performance"
    description = "Runtime efficiency / FPS benchmarking"

    def evaluate(self, context: EvaluationContext) -> Dict[str, Any]:
        dataset = get_dataset(context)
        seeds = context.seeds or [0]
        num_requested = max(1, context.extra_kwargs.get("num_trajectories", 600))
        # Use configurable max horizon for performance benchmarking (default 128 for speed)
        max_perf_horizon = context.extra_kwargs.get("performance_horizon", 128)
        fps_horizon = min(max(context.prediction_lengths), max_perf_horizon) if max_perf_horizon > 0 else max(context.prediction_lengths)
        available = dataset.available_windows(fps_horizon)
        if available <= 0:
            return {
                "status": "skipped",
                "reason": "insufficient_data",
            }

        sample_total = min(num_requested, available)
        per_seed = max(1, math.ceil(sample_total / len(seeds)))

        results = {}
        for step in context.sample_steps:
            metrics = self._benchmark_step(
                context=context,
                dataset=dataset,
                horizon=fps_horizon,
                sample_step=step,
                per_seed=per_seed,
                sample_total=sample_total,
                seeds=seeds,
            )
            results[f"steps_{step}"] = metrics

        return {
            "status": "ok",
            "horizon": fps_horizon,
            "results": results,
        }

    def _benchmark_step(
        self,
        context: EvaluationContext,
        dataset: TrajectoryDataset,
        horizon: int,
        sample_step: int,
        per_seed: int,
        sample_total: int,
        seeds: List[int],
        warmup_rollouts: int = 1,
    ) -> Dict[str, Any]:
        total_frames = 0
        total_time = 0.0
        processed = 0
        warmup_remaining = warmup_rollouts
        measured_rollouts = 0

        for seed in seeds:
            remaining = sample_total - processed
            if remaining <= 0:
                break
            windows = dataset.sample_windows(horizon, min(per_seed, remaining), seed)
            processed += len(windows)
            for window in windows:
                if torch.cuda.is_available() and context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                start = time.perf_counter()
                preds = generate_rollout(
                    model=context.model,
                    vae=context.vae,
                    init_frame=window.init_frame,
                    actions=window.actions,
                    sample_step=sample_step,
                )
                if torch.cuda.is_available() and context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                elapsed = time.perf_counter() - start
                frame_count = preds.shape[0]
                if frame_count == 0:
                    continue
                if warmup_remaining > 0:
                    warmup_remaining -= 1
                    continue
                total_frames += frame_count
                total_time += elapsed
                measured_rollouts += 1

        if total_frames == 0 or total_time == 0:
            return {
                "status": "skipped",
                "reason": "no_measurements",
            }

        fps = total_frames / total_time
        latency = total_time / total_frames
        return {
            "status": "ok",
            "frames": total_frames,
            "rollouts": measured_rollouts,
            "total_time_sec": total_time,
            "fps": fps,
            "avg_frame_latency_sec": latency,
        }
