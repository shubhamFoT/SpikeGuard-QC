"""Turn raw detections into the per-frame facts the decision engine needs."""
from dataclasses import dataclass
from typing import List, Optional
from config import CFG

V = CFG.vision


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: List[float]          # [x1, y1, x2, y2] pixels


def severity_from_area(area_frac: float) -> str:
    if area_frac <= 0:
        return "none"
    if area_frac < V.minor_area:
        return "minor"
    if area_frac < V.major_area:
        return "major"
    return "critical"


def grade(dets: List[Detection], width: int, height: int):
    """Return (primary_class, confidence, bbox, area_frac, severity, decision).

    - primary = the highest-confidence detection above conf_min
    - area_frac = union-free upper bound: sum of box areas, capped at 1 (conservative: it can
      only over-state severity, which errs on the safe side for quality control)
    - decision is deterministic: reject / review / pass. Llama never touches this."""
    dets = [d for d in dets if d.conf >= V.conf_min]
    if not dets:
        return None, 0.0, None, 0.0, "none", "pass"
    primary = max(dets, key=lambda d: d.conf)
    frame_area = float(width * height)
    area = 0.0
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_frac = min(area / frame_area, 1.0)
    sev = severity_from_area(area_frac)
    decision = "reject" if primary.conf >= V.reject_conf else "review"
    return primary.cls, primary.conf, primary.bbox, area_frac, sev, decision
