"""Download a MedMNIST dataset to the local data/ directory.

Usage:
    python scripts/download_data.py [dataset] [--size {28,64,128,224}]

If no dataset is given on the command line, you'll be prompted to pick one
from a menu. The data files are git-ignored; run this script once per
dataset after cloning.
"""

import argparse
from pathlib import Path

import medmnist
from medmnist import INFO

DATASET_NAMES = sorted(INFO.keys())
SPLITS = ("train", "val", "test")


def prompt_for_dataset() -> str:
    print("Available MedMNIST datasets:\n")
    for i, name in enumerate(DATASET_NAMES, start=1):
        info = INFO[name]
        print(f"  {i:2d}. {name} — {info['python_class']}, task: {info['task']}")

    while True:
        choice = input(f"\nSelect a dataset [1-{len(DATASET_NAMES)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(DATASET_NAMES):
            return DATASET_NAMES[int(choice) - 1]
        if choice in DATASET_NAMES:
            return choice
        print("Invalid choice, try again.")


def download(dataset_name: str, data_dir: Path, size: int) -> None:
    info = INFO[dataset_name]
    dataset_class = getattr(medmnist, info["python_class"])

    available_sizes = getattr(dataset_class, "available_sizes", [28])
    if size not in available_sizes:
        raise SystemExit(
            f"Size {size} not available for {info['python_class']}. "
            f"Choose one of {available_sizes}."
        )

    n_classes = len(info["label"])
    print(f"{info['python_class']} — {n_classes} classes, task: {info['task']}")
    print(f"Downloading splits {SPLITS} at size {size}x{size} → {data_dir}\n")

    data_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"  Fetching {split} split …")
        dataset_class(split=split, download=True, size=size, root=str(data_dir))

    print("\nDone. Files saved to:", data_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a MedMNIST dataset")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=DATASET_NAMES,
        default=None,
        help="Dataset to download, e.g. pneumoniamnist (default: prompt interactively)",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[28, 64, 128, 224],
        default=64,
        help="Image resolution to download (default: 64)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to save data (default: data/<dataset_name>/)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset or prompt_for_dataset()
    data_dir = args.data_dir or (
        Path(__file__).parent.parent / "data" / dataset_name
    )
    download(dataset_name, data_dir, args.size)


if __name__ == "__main__":
    main()
