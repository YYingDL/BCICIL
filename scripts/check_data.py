"""Validate the expected local pickle layout without loading full arrays."""

from pathlib import Path
import argparse


REQUIRED = ("x_train.pkl", "state_train.pkl", "x_test.pkl", "state_test.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/saved")
    parser.add_argument("--dataset", choices=("seed3", "mi", "benchmark"), required=True)
    args = parser.parse_args()
    dataset_dir = Path(args.data_root) / args.dataset
    missing = [name for name in REQUIRED if not (dataset_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Missing files under {dataset_dir}: {', '.join(missing)}")
    print(f"OK: {dataset_dir}")


if __name__ == "__main__":
    main()
