from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

import torch

import config.configTrain as cfg
from algorithm import Algorithm
from infer_test import remove_orig_mod_prefix
from models.vae.autoencoder import AutoencoderKL
from metrics import EvaluationContext, MetricRegistry
from metrics.mechanics import MechanicsAccuracyEvaluator
from metrics.performance import InteractionEfficiencyEvaluator
from metrics.visual import VisualQualityEvaluator


def parse_int_list(values: Sequence[str]) -> List[int]:
    result: List[int] = []
    for value in values:
        if value.strip():
            result.append(int(value))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the diffusion-based game-generation model"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(cfg.file_path),
        help="Path to evaluation dataset or manifest",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=600,
        help="Number of rollout windows sampled per horizon",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Subset of metrics to run (default: all registered)",
    )
    parser.add_argument(
        "--prediction-lengths",
        nargs="*",
        default=["1", "16", "64", "256", "1024"],
        help="Prediction horizons to evaluate",
    )
    parser.add_argument(
        "--sample-steps",
        nargs="*",
        default=["2", "4", "6"],
        help="Diffusion sampling steps to benchmark",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=["0", "1", "2"],
        help="Random seeds for repeated rollouts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for metric computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device (use 'auto' to pick CUDA when available)",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=None,
        help="Override path for diffusion checkpoint",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=Path,
        default=None,
        help="Override path for VAE checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./evaluation_results.json"),
        help="File to store aggregated metric results",
    )
    parser.add_argument(
        "--vam-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for the Video-Action Model used in mechanics metrics",
    )
    parser.add_argument(
        "--vam-window",
        type=int,
        default=17,
        help="Temporal window (frames) consumed by the Video-Action Model",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit",
    )
    parser.add_argument(
        "--pretty-print",
        action="store_true",
        help="Pretty-print JSON results to stdout",
    )
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device(cfg.device)
        return torch.device("cpu")
    return torch.device(name)


def load_models(device: torch.device, model_ckpt: Path | None, vae_ckpt: Path | None):
    model = Algorithm(cfg.model_name, device)
    ckpt_path = model_ckpt or Path("ckpt") / cfg.model_path
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_ckpt = remove_orig_mod_prefix(state_dict["network_state_dict"])
    model.load_state_dict(model_ckpt, strict=False)
    model.eval().to(device)

    vae = AutoencoderKL().eval().to(device)
    vae_path = vae_ckpt or Path(cfg.vae_model)
    if vae_path.exists():
        vae_state_dict = torch.load(vae_path, map_location=device, weights_only=False)
        vae_ckpt = remove_orig_mod_prefix(vae_state_dict["network_state_dict"])
        vae.load_state_dict(vae_ckpt, strict=False)
    else:
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
    return model, vae


def build_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(VisualQualityEvaluator())
    registry.register(MechanicsAccuracyEvaluator())
    registry.register(InteractionEfficiencyEvaluator())
    return registry


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    registry = build_registry()
    if args.list_metrics:
        print("Available metrics:")
        for name in registry.registered_names():
            print(f" - {name}")
        return

    device = resolve_device(args.device)
    model, vae = load_models(device, args.model_ckpt, args.vae_ckpt)

    prediction_lengths = parse_int_list(args.prediction_lengths)
    sample_steps = parse_int_list(args.sample_steps)
    seeds = parse_int_list(args.seeds)

    context = EvaluationContext(
        model=model,
        vae=vae,
        device=device,
        dataset_path=args.dataset,
        prediction_lengths=prediction_lengths,
        sample_steps=sample_steps,
        seeds=seeds,
        batch_size=args.batch_size,
        output_dir=args.output.parent.resolve(),
        extra_kwargs={
            "num_trajectories": args.num_trajectories,
            "vam_checkpoint": args.vam_checkpoint,
            "vam_window": args.vam_window,
        },
    )

    context.output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = args.metrics or registry.registered_names()
    results = registry.run(metric_names, context)

    if args.pretty_print:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(json.dumps(results))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
