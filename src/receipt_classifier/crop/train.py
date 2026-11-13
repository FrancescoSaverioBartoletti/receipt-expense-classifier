"""Train a YOLO11-seg model to segment receipts, using configs/crop_train.yaml."""

from pathlib import Path

CONFIG_PATH = Path("configs/crop_train.yaml")
MODELS_DIR = Path("models")


def load_config(path: Path) -> dict:
    """Read the training config."""
    raise NotImplementedError


def train(config: dict) -> Path:
    """Run training and return the path of the best weights."""
    raise NotImplementedError


def export_best(weights: Path, dest_dir: Path) -> Path:
    """Copy the best weights into models/ under a stable name."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
