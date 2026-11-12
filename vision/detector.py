"""Thin wrapper so the rest of the system never imports ultralytics directly.
`weights` may be a .pt file (development on a laptop) or an exported *_executorch_model folder
(the Pi). Swapping between them is the only change needed to move from laptop to device."""
from typing import List, Tuple

from config import CFG
from vision.severity import Detection


class Detector:
    def __init__(self, weights: str, imgsz: int = None, conf: float = None):
        from ultralytics import YOLO
        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz or CFG.vision.imgsz
        self.conf = conf if conf is not None else CFG.vision.conf_min

    def infer(self, bgr) -> Tuple[List[Detection], Tuple[int, int]]:
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        h, w = r.orig_shape
        dets = []
        if r.boxes is not None and len(r.boxes):
            for b, p, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy()):
                dets.append(Detection(r.names[int(c)], float(p), [float(v) for v in b]))
        return dets, (w, h)
