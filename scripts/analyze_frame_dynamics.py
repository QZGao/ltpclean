"""Analyze how much frame-to-frame change exists in a dataset."""
import argparse
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import read_file


def analyze_dynamics(file_path: Path):
    """Check frame-to-frame differences in a recording."""
    data = read_file(file_path)
    
    # Assuming 256x256 RGB format: 256*256*3 + 1 action = 196609 columns
    # Or 128x128: 128*128*3 + 1 = 49153 columns
    num_cols = data.shape[1]
    
    if num_cols == 196609:
        resolution = 256
    elif num_cols == 49153:
        resolution = 128
    else:
        print(f"Unknown format: {num_cols} columns")
        return
    
    num_pixels = resolution * resolution * 3
    frames = data[:, :num_pixels].astype(np.float32) / 127.5 - 1.0  # Convert to [-1, 1]
    frames = frames.reshape(-1, 3, resolution, resolution)
    
    print(f"File: {file_path.name}")
    print(f"Total frames: {len(frames)}")
    print(f"Resolution: {resolution}x{resolution}")
    
    if len(frames) < 2:
        print("Not enough frames to compute differences")
        return
    
    # Compute frame-to-frame differences
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i+1] - frames[i]).mean()
        diffs.append(diff)
    
    diffs = np.array(diffs)
    
    print(f"\nFrame-to-frame change statistics:")
    print(f"  Mean: {diffs.mean():.6f}")
    print(f"  Median: {np.median(diffs):.6f}")
    print(f"  Min: {diffs.min():.6f}")
    print(f"  Max: {diffs.max():.6f}")
    print(f"  Std: {diffs.std():.6f}")
    
    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    print(f"\nPercentage of frames with change > threshold:")
    for thresh in thresholds:
        pct = (diffs > thresh).mean() * 100
        count = (diffs > thresh).sum()
        print(f"  > {thresh:.3f}: {pct:5.1f}% ({count}/{len(diffs)} pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path, help="frameArray files to analyze")
    args = parser.parse_args()
    
    for file_path in args.files:
        analyze_dynamics(file_path)
        print("\n" + "="*60 + "\n")
