"""Convert the Roboflow COCO export into a single-class YOLO-seg dataset.

Input:  data/raw/receipt_cropping.v3i.coco.zip
Output: data/processed/receipt_seg/{train,valid,test}/{images,labels} + dataset.yaml
"""

from pathlib import Path

RAW_ZIP = Path("data/raw/receipt_cropping.v3i.coco.zip")
OUT_DIR = Path("data/processed/receipt_seg")
SPLITS = ("train", "valid", "test")
CLASS_NAME = "receipt"


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Unzip the Roboflow export into a working folder and return its path."""
    raise NotImplementedError


def load_coco(annotations_path: Path) -> dict:
    """Load one split's _annotations.coco.json."""
    raise NotImplementedError


def filter_annotations(coco: dict) -> list[dict]:
    """Keep polygon annotations only: drop box-only labels and duplicates."""
    raise NotImplementedError


def polygon_to_yolo(polygon: list[float], width: int, height: int) -> str:
    """Turn a COCO polygon into a YOLO-seg label line (class 0, normalized xy)."""
    raise NotImplementedError


def convert_split(split: str, src: Path, dest: Path) -> None:
    """Copy images and write YOLO-seg labels for one split."""
    raise NotImplementedError


def write_dataset_yaml(dest: Path) -> None:
    """Write the dataset.yaml that ultralytics reads."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
