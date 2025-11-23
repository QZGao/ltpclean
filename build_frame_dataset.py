from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn

from config import configTrain as cfg

FRAME_RE = re.compile(r"_f(\d+)")
ACTION_RE = re.compile(r"_a(\d+)")


def _extract_number(pattern: re.Pattern[str], name: str, default: int = 0) -> int:
    match = pattern.search(name)
    if match:
        return int(match.group(1))
    return default


def _load_image(path: Path, resolution: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.NEAREST)
    return np.array(img, dtype=np.uint8)


def convert_directory(input_dir: Path, output_path: Path, resolution: int) -> None:
    files = sorted(input_dir.glob("*.png"), key=lambda p: _extract_number(FRAME_RE, p.name))
    if not files:
        raise RuntimeError(f"No PNG files found under {input_dir}")

    rows: List[np.ndarray] = []
    with Progress(
        TextColumn("Converting frames..."),
        BarColumn(bar_width=None),
        "{task.completed}/{task.total}",
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task_id: TaskID = progress.add_task("frames", total=len(files))
        for path in files:
            frame = _load_image(path, resolution)
            action = _extract_number(ACTION_RE, path.name)
            flat = frame.flatten()
            row = np.concatenate([flat, np.array([action], dtype=np.uint8)])
            rows.append(row)
            progress.advance(task_id)

    data = np.stack(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, data, fmt="%d", delimiter=",")
    print(f"Saved {len(rows)} frames to {output_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a directory of frames into frameArray txt format")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing PNG frames")
    parser.add_argument("--output", type=Path, default=Path(cfg.file_path), help="Destination txt file")
    parser.add_argument("--resolution", type=int, default=cfg.img_size, help="Target square resolution")
    return parser.parse_args()


def main():
    args = parse_args()
    convert_directory(args.input_dir, args.output, args.resolution)


if __name__ == "__main__":
    main()
