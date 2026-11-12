"""Frame sources. ConveyorSim replays dataset images as if parts were passing a camera."""
import glob
import itertools
import random
import cv2


class ConveyorSim:
    """Mixes defective and clean images. `defect_prob` can be changed at run time (the demo
    scenarios use this). Be explicit in your report that the 'camera' is a replay of dataset images."""

    def __init__(self, defect_dir, clean_dir=None, defect_prob=0.1, seed=0):
        ext = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.defect = sorted(p for e in ext for p in glob.glob(f"{defect_dir}/{e}"))
        self.clean = sorted(p for e in ext for p in glob.glob(f"{clean_dir}/{e}")) if clean_dir else []
        if not self.defect:
            raise FileNotFoundError(f"no images in {defect_dir}")
        self.defect_prob = defect_prob
        self.rng = random.Random(seed)

    def next(self):
        """Returns (bgr_frame, truth) where truth is 'defect' or 'clean'."""
        use_defect = (not self.clean) or self.rng.random() < self.defect_prob
        pool = self.defect if use_defect else self.clean
        return cv2.imread(self.rng.choice(pool)), ("defect" if use_defect else "clean")


class CameraSource:
    """A real camera (Pi Camera Module 3 via V4L2/libcamera, or a USB webcam)."""

    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def next(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame, "unknown"
