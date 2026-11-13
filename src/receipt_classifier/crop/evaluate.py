"""Evaluate the trained receipt segmentation model on the test split."""

from pathlib import Path

WEIGHTS_PATH = Path("models/receipt_seg_best.pt")
DATASET_YAML = Path("data/processed/receipt_seg/dataset.yaml")


def evaluate(weights: Path, dataset_yaml: Path) -> dict:
    """Run validation on the test split and return mask/box metrics (mAP, IoU)."""
    raise NotImplementedError


def save_predictions(weights: Path, images_dir: Path, out_dir: Path) -> None:
    """Save predicted masks drawn on the test images, for visual inspection."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
