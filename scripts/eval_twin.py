"""Evaluate the SpikeGuard twin: can it tell healthy from faulty windows?
   python -m scripts.eval_twin                                  (synthetic data, runs anywhere)
   python -m scripts.eval_twin --cwru-healthy 97.mat --cwru-fault 105.mat   (real CWRU data)
Reports AUROC on the health index and the zone confusion matrix."""
import argparse
import json

import numpy as np
from sklearn.metrics import confusion_matrix

from config import CFG
from spikeguard_twin.rf_snn import SpikeGuardTwin, estimate_shaft_freq
from spikeguard_twin.signals import load_cwru, synth_vibration, windows
from vision.metrics import binary_report

C = CFG.twin


def get_windows(a):
    if a.cwru_healthy and a.cwru_fault:
        h, rpm = load_cwru(a.cwru_healthy)
        f, _ = load_cwru(a.cwru_fault)
        return list(windows(h, C.fs, C.window_s)), list(windows(f, C.fs, C.window_s)), (rpm or 1797.0)
    h = [synth_vibration(C.fs, C.window_s, 1797, None, seed=1000 + i)[0] for i in range(60)]
    f = [synth_vibration(C.fs, C.window_s, 1797, "BPFO", 1.0, seed=2000 + i)[0] for i in range(60)]
    return h, f, 1797.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwru-healthy")
    ap.add_argument("--cwru-fault")
    a = ap.parse_args()
    h, f, rpm = get_windows(a)
    tw = SpikeGuardTwin(C.fs).calibrate(h[:8], rpm / 60.0)       # calibrate on the first 8 healthy windows
    test = [(w, 0) for w in h[8:]] + [(w, 1) for w in f]
    states = [tw.process(w, estimate_shaft_freq(w, C.fs)) for w, _ in test]
    y = [lab for _, lab in test]
    scores = [s.health_index for s in states]
    rep = binary_report(y, scores, thr=np.median(scores))
    zones_pred = [0 if s.zone == "healthy" else 1 for s in states]
    cm = confusion_matrix(y, zones_pred, labels=[0, 1]).tolist()
    print(json.dumps({"auroc_health_index": rep.get("auroc"), "zone_confusion[healthy,faulty]": cm}, indent=2))


if __name__ == "__main__":
    main()
