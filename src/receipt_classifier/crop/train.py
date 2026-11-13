"""Train a YOLO11-seg model to segment receipts, using configs/crop_train.yaml."""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/crop_train.yaml")
MODELS_DIR = Path("models")
BEST_NAME = "receipt_seg_best.pt"


def load_config(path: Path) -> dict:
    """Read the training config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config: dict) -> tuple[Path, dict]:
    """Run training; return the best weights path and the final validation metrics."""
    params = dict(config)
    model = YOLO(params.pop("model"))
    params["project"] = str(Path(params["project"]).resolve())  # keep runs inside the repo
    metrics = model.train(**params)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    return best, metrics.results_dict


def export_best(weights: Path, dest_dir: Path) -> Path:
    """Copy the best weights into models/ under a stable name."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / BEST_NAME
    shutil.copy2(weights, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="override epochs (e.g. 1 for a quick test)")
    parser.add_argument("--name", help="override the run name")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.name is not None:
        config["name"] = args.name

    best, results = train(config)
    dest = export_best(best, MODELS_DIR)
    print(f"\nBest weights: {best}\nCopied to:    {dest}")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
