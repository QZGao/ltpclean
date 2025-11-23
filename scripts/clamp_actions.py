from argparse import ArgumentParser
from pathlib import Path
import numpy as np

def main():
    parser = ArgumentParser(description="Clamp action IDs in a frameArray txt to a fixed range")
    parser.add_argument("--input", type=Path, required=True, help="Path to the frameArray file")
    parser.add_argument("--max-action", type=int, default=6, help="Maximum inclusive action id")
    args = parser.parse_args()

    path = args.input
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    data = np.loadtxt(path, delimiter=",", dtype=np.int32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    actions = np.clip(data[:, -1], 0, args.max_action)
    data[:, -1] = actions
    np.savetxt(path, data, fmt="%d", delimiter=",")
    unique = np.unique(actions)
    print(f"Clamped {path.name} to {len(unique)} action ids (max {args.max_action}).")

if __name__ == "__main__":
    main()
