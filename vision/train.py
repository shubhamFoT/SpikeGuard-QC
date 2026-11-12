"""Train a small YOLO detector on the prepared NEU-DET data.

   python -m vision.train --data data/neu_yolo/neu.yaml --epochs 60

LICENCE NOTE: Ultralytics YOLO is AGPL-3.0. If you publish your repo, publish it under a
compatible licence (AGPL-3.0) and say so in the README, or choose a permissively licensed model."""
import argparse
from config import CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/neu_yolo/neu.yaml")
    ap.add_argument("--model", default="yolo11n.pt")     # smallest model: best fit for a Pi 5 CPU
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)            # e.g. "0" for a GPU, "cpu" otherwise
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, imgsz=CFG.vision.imgsz, epochs=a.epochs, batch=a.batch,
                seed=0, deterministic=True, patience=15, project="runs", name="neu_y11n",
                device=a.device,
                # factory-condition robustness: lighting and blur (grayscale data, so no hue shift)
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, fliplr=0.5, flipud=0.5)

    best = f"runs/neu_y11n/weights/best.pt"
    metrics = YOLO(best).val(data=a.data, split="test", imgsz=CFG.vision.imgsz)
    print("test mAP50:", metrics.box.map50, "precision:", metrics.box.mp, "recall:", metrics.box.mr)


if __name__ == "__main__":
    main()
