"""NEU-DET contains ONLY defective images, so false-positive rate cannot be measured on it.
This script builds approximate 'clean' frames: it finds a defect-free window in each image
(no overlap with any ground-truth box), then mirror-tiles it to full frame size.

   python -m vision.make_clean_frames --yolo data/neu_yolo --split test --out data/clean_test

This is a documented approximation, NOT a real defect-free dataset. State that in your report,
and prefer real negatives (e.g. good samples from MVTec AD / KolektorSDD) if your mentors agree."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def free_window(w, h, boxes_px, win, tries=200, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(tries):
        x = int(rng.integers(0, w - win + 1))
        y = int(rng.integers(0, h - win + 1))
        if all(x + win <= bx1 or x >= bx2 or y + win <= by1 or y >= by2 for bx1, by1, bx2, by2 in boxes_px):
            return x, y
    return None


def mirror_tile(patch, w, h):
    row = np.hstack([patch, patch[:, ::-1]])
    tiled = np.vstack([row, row[::-1]])
    return cv2.resize(tiled, (w, h), interpolation=cv2.INTER_AREA) if tiled.shape[:2] != (h, w) else tiled


def main(yolo_root, split, out, win_frac=0.4, seed=0):
    root, out = Path(yolo_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    made = 0
    for img_path in sorted((root / "images" / split).iterdir()):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = root / "labels" / split / f"{img_path.stem}.txt"
        boxes = []
        if lbl.exists():
            for ln in lbl.read_text().split("\n"):
                if ln.strip():
                    _, cx, cy, bw, bh = map(float, ln.split())
                    boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        win = int(min(w, h) * win_frac)
        pos = free_window(w, h, boxes, win, rng=rng)
        if pos is None:
            continue
        x, y = pos
        cv2.imwrite(str(out / f"clean_{img_path.stem}.png"), mirror_tile(img[y:y + win, x:x + win], w, h))
        made += 1
    print(f"wrote {made} approximate clean frames to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="data/neu_yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/clean_test")
    a = ap.parse_args()
    main(a.yolo, a.split, a.out)
