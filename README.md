# receipt-expense-classifier
Expense classifier from receipt photos (crop, extract, classify)

## Pipeline
1. **Crop** – locate the receipt in the photo and flatten it (YOLO11-seg + perspective warp).
2. **Extract** – read store, date, items and total.
3. **Classify** – assign each expense to a category.

## Setup
Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.8+ PyTorch build).

```
uv sync
uv run python scripts/check_gpu.py
```

## Data
Step 1 is trained on the Roboflow Universe dataset *receipt_cropping* (v3, CC BY 4.0).
Place the COCO export zip in `data/raw/` (not committed).
