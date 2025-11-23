from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import configTrain as cfg
from metrics.data import TrajectoryDataset
from metrics.vam import VideoActionModel


class VAMClipDataset(Dataset):
    """Dataset that exposes sliding clips for action prediction."""

    def __init__(self, base_dataset: TrajectoryDataset, indices: Sequence[int], window: int):
        if window < 2:
            raise ValueError("Window must be >=2 to provide an action label")
        self.base = base_dataset
        self.indices = list(indices)
        self.window = window

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        stop = start + self.window
        frames = self.base.frames[start:stop]
        if frames.shape[0] != self.window:
            raise ValueError("Insufficient frames for requested window; reduce window size")
        clip = frames.permute(1, 0, 2, 3).contiguous()  # [3, T, H, W]
        clip = clip.add(1.0).mul(0.5)  # map from [-1,1] -> [0,1]
        action_index = min(start + self.window - 2, self.base.actions.shape[0] - 1)
        label = self.base.actions[action_index]
        return clip, label


def _split_indices(indices: List[int], val_split: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(indices)
    if not indices:
        return [], []
    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        train_indices, val_indices = val_indices, []
    return train_indices, val_indices


def _build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def _train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(clips)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def _evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(clips)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def _save_checkpoint(path: Path, model: VideoActionModel, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "window": model.window,
        "num_actions": metadata.get("num_actions"),
        "meta": metadata,
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VideoActionModel (VAM) used for mechanics metrics")
    parser.add_argument("--data-path", type=Path, default=Path(cfg.file_path), help="Path to flattened frame/action txt file")
    parser.add_argument("--window", type=int, default=17, help="Temporal window size in frames")
    parser.add_argument("--stride", type=int, default=2, help="Stride between clip start positions")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional cap on total clips before splitting")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of clips reserved for validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("ckpt/vam_model.pt"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = args.amp and device.type == "cuda"
    data_device = torch.device("cpu")

    dataset = TrajectoryDataset(args.data_path, device=data_device)
    clip_indices = dataset.clip_indices(args.window, stride=args.stride)
    if args.max_clips:
        clip_indices = clip_indices[: args.max_clips]
    if not clip_indices:
        raise RuntimeError("Dataset does not provide enough frames for the requested window")

    train_indices, val_indices = _split_indices(clip_indices, args.val_split, args.seed)
    train_dataset = VAMClipDataset(dataset, train_indices, args.window)
    val_dataset = VAMClipDataset(dataset, val_indices, args.window) if val_indices else None

    train_loader = _build_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = (
        _build_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_dataset is not None
        else None
    )

    model = VideoActionModel(num_actions=dataset.num_actions, window=args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, device)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = model.state_dict()
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
            )
        else:
            improved = train_loss < best_val_loss
            if improved:
                best_val_loss = train_loss
                best_state = model.state_dict()
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f}")

    if best_state is None:
        best_state = model.state_dict()

    if best_state is not model.state_dict():
        model.load_state_dict(best_state)

    metadata = {
        "num_actions": dataset.num_actions,
        "window": args.window,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "clip_stride": args.stride,
        "dataset_path": str(args.data_path),
    }
    _save_checkpoint(args.output, model, metadata)
    print(f"Saved VAM checkpoint to {args.output}")


if __name__ == "__main__":
    main()
