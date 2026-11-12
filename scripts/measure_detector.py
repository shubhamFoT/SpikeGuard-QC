import glob, json, sys, cv2
from vision.detector import Detector
from vision.metrics import binary_report

det = Detector(sys.argv[1] if len(sys.argv) > 1 else "runs/neu_y11n/weights/best.pt")
def top_conf(folder):
    return [max([d.conf for d in det.infer(cv2.imread(p))[0]], default=0.0) for p in sorted(glob.glob(folder + "/*.*"))]

pos = top_conf("data/neu_yolo/images/test")
neg = top_conf("data/clean_test")
rep = binary_report([1] * len(pos) + [0] * len(neg), pos + neg, thr=0.5)
print(json.dumps(rep, indent=2))
