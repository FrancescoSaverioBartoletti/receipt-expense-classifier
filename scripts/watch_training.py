"""Show a live progress bar for a YOLO training run, reading its results.csv.

Usage: python scripts/watch_training.py [run_dir]   (default: runs/crop/yolo11s_seg)
"""

import csv
import sys
import time
from pathlib import Path

import yaml

REFRESH_SECONDS = 5
BAR_WIDTH = 40


def read_results(run_dir: Path) -> list[dict]:
    """Rows of results.csv (one per finished epoch), with stripped column names."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def total_epochs(run_dir: Path) -> int:
    with open(run_dir / "args.yaml", encoding="utf-8") as f:
        return int(yaml.safe_load(f)["epochs"])


def render(rows: list[dict], total: int) -> str:
    done = len(rows)
    filled = int(BAR_WIDTH * done / total)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    line = f"[{bar}] epoch {done}/{total}"
    if rows:
        last = rows[-1]
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(M)"]))
        line += (
            f" | mask mAP50 {float(last['metrics/mAP50(M)']):.3f}"
            f" mAP50-95 {float(last['metrics/mAP50-95(M)']):.3f}"
            f" | best mAP50-95 {float(best['metrics/mAP50-95(M)']):.3f} (epoch {best['epoch']})"
        )
    return line


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/crop/yolo11s_seg")
    total = total_epochs(run_dir)
    last_change = time.time()
    seen = -1
    while True:
        rows = read_results(run_dir)
        if len(rows) != seen:
            seen, last_change = len(rows), time.time()
        print("\r" + render(rows, total), end="", flush=True)
        if seen >= total:
            print("\nTraining finished.")
            return
        if time.time() - last_change > 600:
            print("\nNo new epoch for 10 minutes: training finished early (early stopping) or stopped.")
            return
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
