#!/usr/bin/env python3
"""
benchmark_track_a_d.py — honest benchmarks for SpikeGuard-QC (v2)

Differences from the Phase-1 script:
  * NO literal results (28.4 ms/tok, 745 MB) and NO clamped RAM figures. Anything that cannot be
    measured is printed as "NOT MEASURED: <reason>".
  * Vision latency is measured on REAL images (pre / inference / post split), not a random tensor.
  * Optional: mAP of the .pte itself (not just best.pt) via Ultralytics, and a PyTorch baseline.
  * Llama: real runs of the ExecuTorch runner; tokens/s from the runner's own stats when available.
  * Records the machine (CPU model etc.) so a host number is never mistaken for an Arm-SoC number.

Usage:
  python benchmark_track_a_d.py --images data/test_images --n 100
  python benchmark_track_a_d.py --images data/test_images --pt runs/neu_y11n/weights/best.pt \
         --val-data NEU-DET.yaml --llama-runs 5
"""
import argparse
import json
import platform
import resource
import statistics as st
import time
from pathlib import Path

import cv2
import numpy as np

import edge_pipeline_runner as epr


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def summarize(xs):
    return {"n": len(xs), "p50": round(pct(xs, 50), 2), "p95": round(pct(xs, 95), 2), "mean": round(st.mean(xs), 2)}


def machine_info():
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware", "model\t")):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    import os
    return {"platform": platform.platform(), "cpu": cpu, "arch": platform.machine(),
            "cores": os.cpu_count(), "python": platform.python_version(),
            "note": "HOST measurement. Do not present as Arm Cortex-A SoC performance."}


def file_mb(p):
    p = Path(p)
    return round(p.stat().st_size / 2**20, 2) if p.exists() else None


def bench_vision(cfg, images, n, warmup):
    vs = epr.ExecuTorchVisionStage(cfg.vision_pte, cfg.imgsz, cfg.conf_thr, cfg.iou_thr)
    imgs = [cv2.imread(str(p)) for p in images]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise RuntimeError("no readable images")
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024      # MB on Linux
    for i in range(warmup):
        vs.infer(imgs[i % len(imgs)])
    pre, inf, post, tot, ndet = [], [], [], [], []
    for i in range(n):
        r = vs.infer(imgs[i % len(imgs)])
        pre.append(r.pre_ms); inf.append(r.infer_ms); post.append(r.post_ms); tot.append(r.total_ms); ndet.append(len(r.detections))
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"model": str(cfg.vision_pte), "size_mb": file_mb(cfg.vision_pte), "images": len(imgs),
            "preprocess_ms": summarize(pre), "inference_ms": summarize(inf), "postprocess_ms": summarize(post),
            "end_to_end_ms": summarize(tot), "mean_detections_per_image": round(st.mean(ndet), 2),
            "peak_rss_mb_process": round(rss1, 1), "rss_growth_mb": round(rss1 - rss0, 1)}


def bench_pytorch(pt, images, n):
    try:
        from ultralytics import YOLO
    except ImportError:
        return "NOT MEASURED: ultralytics not installed"
    m = YOLO(str(pt))
    imgs = [cv2.imread(str(p)) for p in images][:n]
    for i in imgs[:5]:
        m.predict(i, imgsz=256, verbose=False)
    ts = []
    for i in imgs:
        t = time.perf_counter(); m.predict(i, imgsz=256, verbose=False); ts.append((time.perf_counter() - t) * 1e3)
    return {"model": str(pt), "size_mb": file_mb(pt),
            "checkpoint_precision_note": "Ultralytics stores best.pt in FP16 - do NOT label it FP32",
            "end_to_end_ms": summarize(ts)}


def val_pte_map(cfg, data_yaml):
    try:
        from ultralytics import YOLO
        res = YOLO(str(cfg.vision_pte.parent)).val(data=data_yaml, imgsz=cfg.imgsz, split="test", verbose=False)
        return {"mAP50": round(float(res.box.map50) * 100, 2), "mAP50_95": round(float(res.box.map) * 100, 2),
                "precision": round(float(res.box.mp) * 100, 2), "recall": round(float(res.box.mr) * 100, 2)}
    except Exception as e:                                       # noqa: BLE001
        return f"NOT MEASURED: {type(e).__name__}: {str(e)[:160]}. Alternative: compare .pte vs .pt outputs on the test set."


def bench_llama(cfg, runs):
    adv = epr.LlamaAdvisor(cfg)
    ok, msg = adv.availability()
    if not ok:
        return f"NOT MEASURED: {msg}"
    kb = epr.load_kb(cfg.kb_path)
    rx = epr.GowinSNNReceiver("replay", cfg.replay_csv)
    cases = [("vibration", epr.Detection(4, "rolled-in_scale", 0.88, (0.2, 0.2, 0.6, 0.6)), "STOP_LINE"),
             ("all_channels", epr.Detection(1, "inclusion", 0.91, (0.1, 0.3, 0.5, 0.7)), "STOP_LINE"),
             ("idle", epr.Detection(2, "patches", 0.72, (0.3, 0.3, 0.5, 0.5)), "QUARANTINE")]
    lat, tps, toks, fmt_ok, est = [], [], [], 0, False
    for k in range(runs):
        sc, det, disp = cases[k % len(cases)]
        pkt = rx.poll(sc)
        vis = epr.VisionResult([det], epr.severity_score([det]), 0, 0, 0, "bench")
        r = adv.advise(pkt, vis, disp, kb)
        if not r.available or r.error and not r.raw_text:
            return f"NOT MEASURED: {r.error}"
        lat.append(r.latency_ms); tps.append(r.tok_per_s); toks.append(r.gen_tokens)
        fmt_ok += int(r.ok); est = est or r.tokens_estimated
    return {"model": str(cfg.llama_pte), "size_mb": file_mb(cfg.llama_pte), "runs": runs,
            "total_response_ms": summarize(lat), "decode_tokens_per_s": summarize(tps),
            "tokens_per_response_mean": round(st.mean(toks), 1), "tokens_estimated": est,
            "format_compliance": f"{fmt_ok}/{runs} responses followed ROOT_CAUSE/ACTION",
            "per_token_ms_p50": round(1000 / pct(tps, 50), 1) if pct(tps, 50) > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with NEU-DET TEST images")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pt", default=None, help="best.pt for a PyTorch baseline")
    ap.add_argument("--val-data", default=None, help="Ultralytics data yaml to validate the .pte on the test split")
    ap.add_argument("--llama-runs", type=int, default=3, help="0 to skip")
    ap.add_argument("--out", default="docs/benchmarks/results.json")
    a = ap.parse_args()

    cfg = epr.Config()
    imgs = sorted(p for p in Path(a.images).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    res = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine_info()}
    try:
        res["stage1_executorch"] = bench_vision(cfg, imgs, a.n, a.warmup)
    except Exception as e:                                       # noqa: BLE001
        res["stage1_executorch"] = f"NOT MEASURED: {type(e).__name__}: {e}"
    if a.pt:
        res["stage1_pytorch_baseline"] = bench_pytorch(a.pt, imgs, min(a.n, len(imgs)))
    res["stage1_pte_accuracy"] = val_pte_map(cfg, a.val_data) if a.val_data else "NOT MEASURED: pass --val-data"
    res["stage2_llama"] = bench_llama(cfg, a.llama_runs) if a.llama_runs else "NOT MEASURED: skipped"

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
