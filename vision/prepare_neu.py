"""Convert NEU-DET (Pascal VOC XML boxes) to the YOLO folder layout and split it.

  python -m vision.prepare_neu --src /path/to/NEU-DET --out data/neu_yolo

Tolerant on purpose: it searches the source tree for every .xml and finds the matching image
by file name, because NEU-DET zips come in several folder layouts."""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from config import CLASS_NAMES

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((name, x1, y1, x2, y2))
    return w, h, boxes


def to_yolo_lines(w, h, boxes):
    lines = []
    for name, x1, y1, x2, y2 in boxes:
        if name not in CLASS_NAMES:
            raise ValueError(f"unknown class {name!r}; expected one of {CLASS_NAMES}")
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{CLASS_NAMES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def find_image(src, stem):
    for p in src.rglob(f"{stem}*"):
        if p.suffix.lower() in IMG_EXT:
            return p
    return None


def main(src, out, seed=0, split=(0.70, 0.15, 0.15)):
    src, out = Path(src), Path(out)
    items = []
    for xml in sorted(src.rglob("*.xml")):
        img = find_image(src, xml.stem)
        if img is not None:
            items.append((xml, img))
    if not items:
        raise SystemExit("no (xml, image) pairs found - check --src")
    random.Random(seed).shuffle(items)
    n = len(items)
    cuts = {"train": items[: int(split[0] * n)],
            "val": items[int(split[0] * n): int((split[0] + split[1]) * n)],
            "test": items[int((split[0] + split[1]) * n):]}
    for name, subset in cuts.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for xml, img in subset:
            w, h, boxes = parse_voc(xml)
            shutil.copy(img, out / "images" / name / img.name)
            (out / "labels" / name / f"{img.stem}.txt").write_text("\n".join(to_yolo_lines(w, h, boxes)))
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASS_NAMES))
    (out / "neu.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    print({k: len(v) for k, v in cuts.items()}, "->", out / "neu.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/neu_yolo")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.src, a.out, a.seed)
