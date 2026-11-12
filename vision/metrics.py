"""Metrics the Arm mentors asked for: precision, recall, F1, FPR, FNR, AUROC, AUPRC,
latency, FPS (and memory, read from /proc so it works on the Pi without extra packages)."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)


def binary_report(y_true, y_score, thr=0.5):
    """y_true: 1 = defective, 0 = good.  y_score: model's defect score/confidence."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), precision=prec, recall=rec, f1=f1,
               fpr=fp / (fp + tn) if fp + tn else float("nan"),
               fnr=fn / (fn + tp) if fn + tp else float("nan"))
    if len(set(y_true.tolist())) == 2:      # AUROC/AUPRC need both classes present
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    return out


def latency_stats(ms):
    a = np.asarray(ms, dtype=float)
    return dict(n=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), fps=float(1000.0 / a.mean()))


def rss_mb():
    """Resident memory of this process in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")
