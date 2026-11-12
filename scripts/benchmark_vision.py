"""Measure end-to-end vision latency, FPS and memory on the target device.
   python -m scripts.benchmark_vision --weights best_executorch_model --images data/neu_yolo/images/test --n 200
Run it once per variant (.pt on CPU, FP32 ExecuTorch, INT8 ExecuTorch) and paste the rows into the report."""
import argparse
import glob
import json
import time

import cv2

from config import CFG
from vision.detector import Detector
from vision.metrics import latency_stats, rss_mb
from vision.severity import grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="bench_vision.json")
    a = ap.parse_args()
    paths = sorted(glob.glob(f"{a.images}/*.*"))
    imgs = [cv2.imread(p) for p in paths[: a.n]]
    det = Detector(a.weights)
    for im in imgs[: a.warmup]:                     # first calls are slow (allocation, cache): discard
        det.infer(im)
    times = []
    for im in imgs:
        t = time.perf_counter()
        dets, (w, h) = det.infer(im)
        grade(dets, w, h)                           # include post-processing: this is END-TO-END
        times.append((time.perf_counter() - t) * 1000)
    res = {"weights": a.weights, "imgsz": CFG.vision.imgsz, **latency_stats(times), "rss_mb": rss_mb()}
    print(json.dumps(res, indent=2))
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
