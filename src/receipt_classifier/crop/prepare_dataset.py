"""Convert the Roboflow COCO export into a single-class YOLO-seg dataset.

Input:  the zip set in configs/dataset.yaml
Output: data/processed/receipt_seg/{train,valid,test}/{images,labels}
"""

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path("configs/dataset.yaml")
ANNOTATIONS_FILE = "_annotations.coco.json"
CLASS_ID = 0           # single class: receipt
DEDUP_IOU = 0.9        # two polygons overlapping more than this are the same receipt


def load_config(path: Path) -> dict:
    """Read the dataset config and turn its paths into Path objects."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["raw_zip"] = Path(cfg["raw_zip"])
    cfg["extract_dir"] = Path(cfg["extract_dir"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    return cfg


def load_exclusions(cfg: dict) -> set[str]:
    """File names to skip: the inline `exclude` list plus every `exclude_files` list."""
    names = set(cfg.get("exclude") or [])
    for path in cfg.get("exclude_files") or []:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line)
    return names


def is_extracted(extract_dir: Path, splits: list[str]) -> bool:
    """True if every split folder already has its COCO annotations file."""
    return all((extract_dir / split / ANNOTATIONS_FILE).is_file() for split in splits)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Unzip the Roboflow export into extract_dir."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)


def load_coco(json_path: Path) -> dict:
    """Load one split's _annotations.coco.json."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def group_by_image(coco: dict) -> dict[int, list[dict]]:
    """Map each image id to its annotations (images without any get an empty list)."""
    groups = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        groups[ann["image_id"]].append(ann)
    return groups


def main_polygon(ann: dict) -> list[float] | None:
    """Return the largest polygon of an annotation, or None if it is box-only."""
    polygons = [p for p in ann.get("segmentation") or [] if len(p) >= 6]  # >= 3 points
    if not polygons:
        return None
    return max(polygons, key=lambda p: cv2.contourArea(to_points(p)))


def to_points(polygon: list[float]) -> np.ndarray:
    """Flat [x1, y1, x2, y2, ...] -> (N, 2) int32 array for OpenCV."""
    return np.round(np.array(polygon, dtype=np.float32).reshape(-1, 2)).astype(np.int32)


def polygon_iou(a: list[float], b: list[float], width: int, height: int) -> float:
    """Overlap between two polygons, computed on rasterized masks."""
    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_a, [to_points(a)], 1)
    cv2.fillPoly(mask_b, [to_points(b)], 1)
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.0


def clean_annotations(anns: list[dict], width: int, height: int) -> tuple[list[list[float]], Counter]:
    """Keep one polygon per receipt: drop box-only labels and duplicates."""
    stats = Counter()
    kept: list[list[float]] = []
    for ann in anns:
        polygon = main_polygon(ann)
        if polygon is None:
            stats["box_only"] += 1
        elif any(polygon_iou(polygon, k, width, height) > DEDUP_IOU for k in kept):
            stats["duplicate"] += 1
        else:
            kept.append(polygon)
    return kept, stats


def polygon_to_yolo(polygon: list[float], width: int, height: int) -> str:
    """Turn a COCO polygon into a YOLO-seg label line (class 0, normalized xy)."""
    points = np.array(polygon, dtype=np.float64).reshape(-1, 2) / (width, height)
    points = np.clip(points, 0.0, 1.0)
    return f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in points.ravel())


def convert_split(split: str, src: Path, dest: Path, exclude: set[str]) -> Counter:
    """Copy images and write YOLO-seg labels for one split; return the counts."""
    coco = load_coco(src / split / ANNOTATIONS_FILE)
    images_dir, labels_dir = dest / split / "images", dest / split / "labels"
    if (dest / split).exists():
        shutil.rmtree(dest / split)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    stats = Counter()
    images = {img["id"]: img for img in coco["images"]}
    for image_id, anns in group_by_image(coco).items():
        img = images[image_id]
        if img["file_name"] in exclude:
            stats["excluded"] += 1
            continue
        polygons, ann_stats = clean_annotations(anns, img["width"], img["height"])
        stats.update(ann_stats)
        if not polygons:
            # An unlabeled receipt would teach the model it is background.
            stats["images_dropped"] += 1
            continue

        shutil.copy2(src / split / img["file_name"], images_dir / img["file_name"])
        lines = [polygon_to_yolo(p, img["width"], img["height"]) for p in polygons]
        (labels_dir / Path(img["file_name"]).with_suffix(".txt").name).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        stats["images_kept"] += 1
        stats["polygons_kept"] += len(polygons)
    return stats


def write_dataset_yaml(dest: Path, class_name: str) -> Path:
    """Write the dataset.yaml that ultralytics reads (absolute path, one class)."""
    content = {
        "path": dest.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {CLASS_ID: class_name},
    }
    yaml_path = dest / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)
    return yaml_path


def print_report(all_stats: dict[str, Counter]) -> None:
    """Print kept/dropped counts per split."""
    header = (
        f"{'split':<6} {'images kept':>11} {'img dropped':>11} {'excluded':>8} "
        f"{'polygons':>8} {'box-only':>8} {'duplicates':>10}"
    )
    print(header)
    print("-" * len(header))
    for split, s in all_stats.items():
        print(
            f"{split:<6} {s['images_kept']:>11} {s['images_dropped']:>11} {s['excluded']:>8} "
            f"{s['polygons_kept']:>8} {s['box_only']:>8} {s['duplicate']:>10}"
        )


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    extract_dir, splits = cfg["extract_dir"], cfg["splits"]

    if is_extracted(extract_dir, splits):
        print(f"Already extracted: {extract_dir}")
    else:
        print(f"Extracting {cfg['raw_zip']} -> {extract_dir}")
        extract_zip(cfg["raw_zip"], extract_dir)

    exclude = load_exclusions(cfg)
    all_stats = {split: convert_split(split, extract_dir, cfg["output_dir"], exclude) for split in splits}
    yaml_path = write_dataset_yaml(cfg["output_dir"], cfg["class_name"])
    print(f"YOLO-seg dataset written to {cfg['output_dir']} ({yaml_path.name})\n")
    print_report(all_stats)


if __name__ == "__main__":
    main()
