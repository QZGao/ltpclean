from argparse import ArgumentParser
from pathlib import Path
import numpy as np

def main():
    parser = ArgumentParser(description="Inspect action distribution in frameArray files")
    parser.add_argument("paths", nargs="+", type=Path, help="One or more frameArray txt files")
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            print(f"{path}: file missing")
            continue
        data = np.loadtxt(path, delimiter=",", dtype=int)
        actions = data[:, -1]
        unique = np.unique(actions)
        print(f"{path.name}")
        print(f"  total frames: {len(actions)}")
        print(f"  max action id: {actions.max()}")
        print(f"  unique actions: {len(unique)} ({unique.tolist()})")

if __name__ == "__main__":
    main()
