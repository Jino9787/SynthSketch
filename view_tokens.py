import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=Path,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.file

    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() != ".npy":
        raise SystemExit(f"Not an .npy file: {path}")

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    tokens = np.load(path, allow_pickle=True)
    print(f"Loaded: {path}")
    print(f"shape: {tokens.shape}")
    print(f"dtype: {tokens.dtype}")
    print()
    for index, row in enumerate(tokens):
        print(f"{index:02d}: {row}")


if __name__ == "__main__":
    main()
