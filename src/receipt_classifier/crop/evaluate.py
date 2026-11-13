"""Evaluate the trained receipt segmentation model on a dataset split.

Writes to runs/crop/eval_<split>/:
  metrics.txt     standard ultralytics metrics + per-image IoU summary
  per_image.csv   IoU and category per image, worst first
  overlays/       each image with the true outline (red) and predicted mask (green)
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_PATH = Path("models/receipt_seg_best.pt")
DATASET_DIR = Path("data/processed/receipt_seg")
RUNS_DIR = Path("runs/crop")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
GOOD_IOU = 0.90
PARTIAL_IOU = 0.50

GT_COLOR = (0, 0, 255)      # red (BGR)
PRED_COLOR = (0, 200, 0)    # green


def load_gt_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    """Draw the true mask from a YOLO-seg label file, at full image resolution."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()[1:]  # drop the class id
        if not values:
            continue
        points = np.array(values, dtype=np.float64).reshape(-1, 2) * (width, height)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


def predict_mask(model: YOLO, image_path: Path, conf: float) -> tuple[np.ndarray | None, float]:
    """Return the most confident predicted mask (full resolution) and its confidence."""
    result = model.predict(str(image_path), conf=conf, retina_masks=True, verbose=False)[0]
    if result.masks is None or len(result.boxes) == 0:
        return None, 0.0
    best = int(result.boxes.conf.argmax())
    mask = result.masks.data[best].cpu().numpy() > 0.5
    return mask.astype(np.uint8), float(result.boxes.conf[best])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def categorize(iou: float) -> str:
    if iou >= GOOD_IOU:
        return "good"
    if iou >= PARTIAL_IOU:
        return "partial"
    return "missed"


def draw_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray | None, iou: float, conf: float) -> np.ndarray:
    """Predicted mask as a green fill, true outline as a red line, IoU written on top."""
    out = image.copy()
    thickness = max(2, image.shape[1] // 300)
    if pred is not None:
        fill = out.copy()
        fill[pred.astype(bool)] = PRED_COLOR
        out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, PRED_COLOR, thickness)
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, GT_COLOR, thickness)

    text = f"IoU {iou:.3f}  conf {conf:.2f}  {categorize(iou)}"
    scale = image.shape[1] / 900
    org = (10, int(40 * scale) + 10)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def run_standard_metrics(weights: Path, split: str, out_dir: Path) -> dict:
    """Ultralytics validation metrics (precision, recall, mAP) on the split."""
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATASET_DIR / "dataset.yaml"), split="val" if split == "valid" else split,
        project=str(out_dir.resolve()), name="ultralytics", exist_ok=True, plots=False, verbose=False,
    )
    return metrics.results_dict


def evaluate_per_image(weights: Path, split: str, conf: float, out_dir: Path) -> list[dict]:
    """IoU of the most confident mask against the true mask, plus an overlay per image."""
    model = YOLO(str(weights))
    images_dir, labels_dir = DATASET_DIR / split / "images", DATASET_DIR / split / "labels"
    overlays_dir = out_dir / "overlays"
    if overlays_dir.exists():
        shutil.rmtree(overlays_dir)
    overlays_dir.mkdir(parents=True)

    rows = []
    for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt = load_gt_mask(labels_dir / f"{image_path.stem}.txt", width, height)
        pred, confidence = predict_mask(model, image_path, conf)
        iou = mask_iou(gt, pred) if pred is not None else 0.0

        # IoU first in the name, so the file explorer lists the worst images on top.
        overlay = draw_overlay(image, gt, pred, iou, confidence)
        cv2.imwrite(str(overlays_dir / f"{iou:.3f}_{image_path.stem}.jpg"), overlay)
        rows.append({"image": image_path.name, "conf": round(confidence, 3), "iou": round(iou, 4),
                     "category": categorize(iou)})
    return sorted(rows, key=lambda r: r["iou"])


def write_report(metrics: dict, rows: list[dict], out_dir: Path) -> None:
    """Write per_image.csv and metrics.txt, and print the summary."""
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ious = np.array([r["iou"] for r in rows])
    lines = ["Standard metrics (mask):"]
    for key in ("precision", "recall", "mAP50", "mAP50-95"):
        lines.append(f"  {key:<10} {metrics[f'metrics/{key}(M)']:.4f}")
    lines.append(f"\nPer image ({len(rows)} images, most confident mask):")
    lines.append(f"  mean IoU   {ious.mean():.4f}")
    lines.append(f"  median IoU {np.median(ious):.4f}")
    for category in ("good", "partial", "missed"):
        n = sum(r["category"] == category for r in rows)
        lines.append(f"  {category:<10} {n:>3} ({n / len(rows):.0%})")
    lines.append("\nWorst 5:")
    lines += [f"  {r['iou']:.3f}  {r['image']}" for r in rows[:5]]

    report = "\n".join(lines)
    (out_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence to accept a mask")
    args = parser.parse_args()

    out_dir = RUNS_DIR / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_standard_metrics(args.weights, args.split, out_dir)
    rows = evaluate_per_image(args.weights, args.split, args.conf, out_dir)
    write_report(metrics, rows, out_dir)
    print(f"\nOverlays: {out_dir / 'overlays'}")


if __name__ == "__main__":
    main()
