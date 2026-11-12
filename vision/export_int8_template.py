"""TEMPLATE: INT8 post-training quantisation through ExecuTorch's PT2E flow (XNNPACK quantizer).

Treat this as a starting point, not a guaranteed recipe: module paths moved between ExecuTorch /
torch releases (imports below try both), and YOLO's detection head may need small changes to
export cleanly. Budget 1-2 days. If it will not export, report FP32 and say why; the Llama model
is already quantised (8-bit activations / 4-bit weights), which still shows the technique.

Steps: load the network -> capture the graph -> insert observers -> calibrate with REAL
images -> convert -> lower to XNNPACK -> write .pte."""
import argparse
import glob

import cv2
import numpy as np
import torch

from config import CFG


def load_images(folder, n, size):
    paths = sorted(glob.glob(f"{folder}/*.*"))[:n]
    for p in paths:
        im = cv2.resize(cv2.imread(p), (size, size))[:, :, ::-1].astype(np.float32) / 255.0
        yield torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calib", required=True, help="folder of ~100 representative images (train split)")
    ap.add_argument("--out", default="yolo_int8.pte")
    a = ap.parse_args()
    size = CFG.vision.imgsz

    from ultralytics import YOLO
    net = YOLO(a.weights).model.fuse().eval().float()      # conv+BN fused, FP32, eval mode
    for m in net.modules():
        if hasattr(m, "export"):
            m.export = True                                # detection head returns a plain tensor
    example = (torch.randn(1, 3, size, size),)

    try:
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    except ImportError:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from executorch.exir import to_edge_transform_and_lower

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    try:
        gm = torch.export.export_for_training(net, example).module()
    except AttributeError:                                  # newer torch: export() is enough
        gm = torch.export.export(net, example).module()
    gm = prepare_pt2e(gm, quantizer)
    with torch.no_grad():
        for x in load_images(a.calib, 100, size):           # calibration = observe real activations
            gm(x)
    gm = convert_pt2e(gm)

    edge = to_edge_transform_and_lower(torch.export.export(gm, example),
                                       partitioner=[XnnpackPartitioner()])
    with open(a.out, "wb") as f:
        f.write(edge.to_executorch().buffer)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
