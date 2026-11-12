"""Export trained weights to ExecuTorch (.pte, XNNPACK backend).

   python -m vision.export_executorch --weights runs/neu_y11n/weights/best.pt

Needs Python 3.10-3.13, torch >= 2.9 and the `executorch` package on the EXPORT machine
(a laptop or Colab is fine). Copy the resulting *_executorch_model/ folder to the Pi.
Ultralytics currently exports FP32; see export_int8_template.py for an INT8 attempt."""
import argparse
from config import CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    out = YOLO(a.weights).export(format="executorch", imgsz=CFG.vision.imgsz)
    print("exported to:", out)


if __name__ == "__main__":
    main()
