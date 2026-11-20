#!/usr/bin/env python3
"""
edge_pipeline_runner.py — SpikeGuard-QC unified 3-stage edge runner  (v2, no simulated inference)

Stage 0  FPGA telemetry   : parses the real 12-byte UART frames (CRC-checked, resync-capable) from
                            (a) a live serial port, or (b) a replay of frames produced by the actual
                            RTL simulation (tb/tb_spikeguard_top.v -> sim/uart_frames.csv).
Stage 1  Vision           : runs the exported YOLO11n ExecuTorch .pte on a REAL image, decodes the
                            (1, 4+nc, N) output, applies NMS, computes a severity score.
Rule engine               : deterministic PASS / QUARANTINE / STOP_LINE decision (never waits for the LLM).
Stage 2  Llama advisor    : Llama-3.2-1B-Instruct (.pte) via the ExecuTorch Llama runner; grounded with
                            offline notes from knowledge/defect_kb.json; writes root-cause + corrective action.
Audit log                 : SQLite, append-only, SHA-256 hash chain (tamper-evident), survives restarts.

Nothing here fabricates a measurement. If a model/runner is missing the stage reports UNAVAILABLE.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# NEU-DET class order used in training (alphabetical). Verify against your data.yaml `names:`.
DEFECT_CLASSES: Tuple[str, ...] = (
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
)

# Illustrative per-class criticality used by the severity score. CALIBRATE with a process engineer.
CLASS_CRITICALITY: Dict[str, float] = {
    "crazing": 0.70, "inclusion": 0.90, "patches": 0.50,
    "pitted_surface": 0.80, "rolled-in_scale": 0.60, "scratches": 0.60,
}

# Physical ceiling of the Stage-0 LIF design: 100 samples / 1 ms window, all channels -> 50 spikes.
SPIKES_PER_WINDOW_MAX = 50
Q88 = 256.0
V_THRESHOLD_AU = 64.0     # LIF firing threshold in the neuron's own (dimensionless) units


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    vision_pte: Path = Path(os.getenv("SPIKEGUARD_VISION_PTE",
                            ROOT / "runs/neu_y11n/weights/best_executorch_model/best.pte"))
    llama_pte: Path = Path(os.getenv("SPIKEGUARD_LLAMA_PTE", ROOT / "models/llama_et/model.pte"))
    llama_tokenizer: Path = Path(os.getenv("SPIKEGUARD_LLAMA_TOKENIZER", ROOT / "models/llama_et/tokenizer.model"))
    llama_runner: str = os.getenv("LLAMA_RUNNER", "llama_main")
    # Placeholders: {model} {tokenizer} {prompt} {seq_len}. Check `llama_main --help` for your ExecuTorch version.
    llama_args: str = os.getenv(
        "LLAMA_RUNNER_ARGS",
        "--model_path {model} --tokenizer_path {tokenizer} --prompt {prompt} --temperature 0 --seq_len {seq_len}")
    llama_seq_len: int = int(os.getenv("LLAMA_SEQ_LEN", "400"))
    llama_timeout_s: int = int(os.getenv("LLAMA_TIMEOUT_S", "240"))
    imgsz: int = 256
    conf_thr: float = 0.35
    iou_thr: float = 0.45
    # rule-engine thresholds (spikes per 1 ms window; hardware max = 50, hw_irq fires at 15)
    irq_spikes: int = 15
    stop_spikes: int = 30
    stop_severity: float = 0.75
    kb_path: Path = ROOT / "knowledge/defect_kb.json"
    db_path: Path = Path(os.getenv("SPIKEGUARD_DB", ROOT / "logs/qc_audit.db"))
    image_dir: Path = ROOT / "logs/images"
    replay_csv: Path = ROOT / "sim/uart_frames.csv"


# --------------------------------------------------------------------------------------
# Stage 0 — FPGA UART telemetry
# --------------------------------------------------------------------------------------
SOF, EOF, FRAME_LEN = 0xA5, 0x5A, 12


def xor_crc(payload: bytes) -> int:
    c = 0
    for b in payload:
        c ^= b
    return c


@dataclass
class FPGASpikePacket:
    trig_flags: int
    spike_count: int
    peak_vm_q88: int
    delta_rate: int
    window_us: int
    source: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def peak_vm(self) -> float:
        """Peak membrane potential in the neuron's arbitrary units (threshold = 64.0). NOT millivolts."""
        return self.peak_vm_q88 / Q88

    @property
    def hw_irq(self) -> bool:
        return self.spike_count >= 15

    @property
    def trigger_sources(self) -> List[str]:
        names = [(0x01, "ACOUSTIC"), (0x02, "VIBRATION"), (0x04, "CURRENT")]
        return [n for m, n in names if self.trig_flags & m] or ["NONE"]


def parse_frame(buf: bytes, source: str = "") -> Optional[FPGASpikePacket]:
    """Return a packet if buf is exactly one valid frame (SOF/EOF/CRC/window ok), else None."""
    if len(buf) != FRAME_LEN or buf[0] != SOF or buf[11] != EOF:
        return None
    if xor_crc(buf[1:10]) != buf[10]:
        return None
    flags = buf[1] & 0x07
    spikes, peak, delta, win = struct_u16(buf, 2), struct_u16(buf, 4), struct_u16(buf, 6), struct_u16(buf, 8)
    return FPGASpikePacket(flags, spikes, peak, delta, win, source=source)


def struct_u16(b: bytes, i: int) -> int:
    return b[i] | (b[i + 1] << 8)


class FrameParser:
    """Streaming parser with resynchronisation: discards garbage / corrupted frames (EMI)."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.bad_frames = 0

    def feed(self, data: bytes, source: str = "") -> List[FPGASpikePacket]:
        self.buf.extend(data)
        out: List[FPGASpikePacket] = []
        while True:
            i = self.buf.find(bytes([SOF]))
            if i < 0:
                self.buf.clear(); break
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < FRAME_LEN:
                break
            pkt = parse_frame(bytes(self.buf[:FRAME_LEN]), source)
            if pkt is not None:
                out.append(pkt); del self.buf[:FRAME_LEN]
            else:
                self.bad_frames += 1; del self.buf[:1]     # slide one byte and resync
        return out


class GowinSNNReceiver:
    """Stage-0 source. mode='replay' (RTL-simulation frames) or mode='serial' (real FPGA)."""

    def __init__(self, mode: str = "replay", replay_csv: Optional[Path] = None,
                 port: str = "/dev/serial0", baud: int = 460800):
        self.mode, self.port, self.baud = mode, port, baud
        self._frames: Dict[str, List[Tuple[int, FPGASpikePacket]]] = {}
        self._ser = None
        self._parser = FrameParser()
        if mode == "replay":
            self._load_replay(Path(replay_csv or Config().replay_csv))
        elif mode == "serial":
            import serial  # pyserial (lazy)
            self._ser = serial.Serial(port, baud, timeout=0.2)
        else:
            raise ValueError("mode must be 'replay' or 'serial'")

    # ---- replay ----
    def _load_replay(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/run_rtl_sim.sh first (needs iverilog).")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name, idx = row["scenario"].strip(), int(row["idx"])
                pkt = parse_frame(bytes.fromhex(row["frame_hex"].strip()),
                                  source=f"RTL simulation (Icarus Verilog) scenario={name} window={idx}")
                if pkt is not None:
                    self._frames.setdefault(name, []).append((idx, pkt))

    def scenarios(self) -> List[str]:
        return [s for s in self._frames if s != "boot"]

    def poll(self, scenario: Optional[str] = None, timeout: float = 2.0) -> FPGASpikePacket:
        if self.mode == "replay":
            if scenario not in self._frames:
                raise KeyError(f"unknown scenario {scenario!r}; have {self.scenarios()}")
            cands = [p for idx, p in self._frames[scenario] if idx >= 1] or [p for _, p in self._frames[scenario]]
            return max(cands, key=lambda p: (p.spike_count, p.peak_vm_q88))   # the most informative window
        deadline = time.time() + timeout
        last: Optional[FPGASpikePacket] = None
        while time.time() < deadline:
            data = self._ser.read(64)
            for pkt in self._parser.feed(data, source=f"UART {self.port} @ {self.baud} 8N1"):
                last = pkt
            if last is not None and not data:
                return last
        if last is None:
            raise TimeoutError(f"no valid 12-byte frame on {self.port} within {timeout}s")
        return last


# --------------------------------------------------------------------------------------
# Stage 1 — Vision (ExecuTorch .pte)
# --------------------------------------------------------------------------------------
@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    box: Tuple[float, float, float, float]        # normalised x1,y1,x2,y2 in ORIGINAL image coords

    @property
    def area_frac(self) -> float:
        return max(0.0, (self.box[2] - self.box[0])) * max(0.0, (self.box[3] - self.box[1]))


@dataclass
class VisionResult:
    detections: List[Detection]
    severity: float
    pre_ms: float
    infer_ms: float
    post_ms: float
    backend: str

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms


def severity_score(dets: Sequence[Detection], area_ref: float = 0.15) -> float:
    """
    0..1 heuristic:  per defect  s = criticality(class) * (0.4*conf + 0.6*min(1, area/area_ref))
    overall = max(s) + 0.05 per additional defect (capped at 1). Calibrate weights with a process engineer.
    """
    if not dets:
        return 0.0
    per = [CLASS_CRITICALITY.get(d.class_name, 0.6) * (0.4 * d.conf + 0.6 * min(1.0, d.area_frac / area_ref))
           for d in dets]
    return float(min(1.0, max(per) + 0.05 * (len(dets) - 1)))


class ExecuTorchVisionStage:
    def __init__(self, pte_path: Path, imgsz: int = 256, conf_thr: float = 0.35, iou_thr: float = 0.45,
                 class_names: Sequence[str] = DEFECT_CLASSES):
        self.pte_path, self.imgsz = Path(pte_path), imgsz
        self.conf_thr, self.iou_thr, self.class_names = conf_thr, iou_thr, tuple(class_names)
        if not self.pte_path.exists():
            raise FileNotFoundError(f"vision model not found: {self.pte_path}")
        try:
            import torch                                   # noqa: F401
            from executorch.runtime import Runtime
        except ImportError as e:                           # ExecuTorch wheels are Linux/macOS -> use WSL2
            raise RuntimeError("executorch/torch not importable. Run inside WSL2/Linux: "
                               "pip install executorch torch") from e
        self._rt = Runtime.get()
        self._method = self._rt.load_program(self.pte_path).load_method("forward")
        self.backend = f"ExecuTorch {self.pte_path.name} ({self.pte_path.stat().st_size / 2**20:.2f} MB, host CPU)"

    # -- pre/post-processing (pure numpy/cv2; unit-tested without ExecuTorch) --
    @staticmethod
    def letterbox(bgr: np.ndarray, imgsz: int) -> Tuple[np.ndarray, float, int, int]:
        h, w = bgr.shape[:2]
        r = min(imgsz / h, imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, r, px, py

    @staticmethod
    def decode(pred: np.ndarray, imgsz: int, ratio: float, pad_x: int, pad_y: int,
               orig_w: int, orig_h: int, class_names: Sequence[str],
               conf_thr: float, iou_thr: float) -> List[Detection]:
        """pred: (1, 4+nc, N) or (4+nc, N) or (N, 4+nc); boxes cx,cy,w,h (pixels or 0-1)."""
        nc = len(class_names)
        p = np.asarray(pred, dtype=np.float32)
        if p.ndim == 3:
            p = p[0]
        if p.shape[0] != 4 + nc and p.shape[1] == 4 + nc:
            p = p.T
        if p.shape[0] != 4 + nc:
            raise ValueError(f"unexpected output shape {pred.shape} for {nc} classes")
        boxes, scores = p[:4].T, p[4:].T
        if scores.max() > 1.0 or scores.min() < 0.0:       # logits -> probabilities
            scores = 1.0 / (1.0 + np.exp(-scores))
        cls, conf = scores.argmax(1), scores.max(1)
        keep = conf >= conf_thr
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if boxes.size == 0:
            return []
        if boxes.max() <= 1.5:                              # normalised -> pixels of the letterboxed input
            boxes = boxes * imgsz
        cx, cy, bw, bh = boxes.T
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        out: List[Detection] = []
        for c in np.unique(cls):                            # class-aware NMS
            m = np.where(cls == c)[0]
            rects = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in m]
            idx = cv2.dnn.NMSBoxes(rects, [float(conf[i]) for i in m], conf_thr, iou_thr)
            for j in np.array(idx).reshape(-1):
                i = m[int(j)]
                bx1 = np.clip((x1[i] - pad_x) / ratio / orig_w, 0, 1)
                by1 = np.clip((y1[i] - pad_y) / ratio / orig_h, 0, 1)
                bx2 = np.clip((x2[i] - pad_x) / ratio / orig_w, 0, 1)
                by2 = np.clip((y2[i] - pad_y) / ratio / orig_h, 0, 1)
                out.append(Detection(int(c), class_names[int(c)], float(conf[i]),
                                     (float(bx1), float(by1), float(bx2), float(by2))))
        return sorted(out, key=lambda d: -d.conf)

    def infer(self, bgr: np.ndarray) -> VisionResult:
        import torch
        t0 = time.perf_counter()
        canvas, r, px, py = self.letterbox(bgr, self.imgsz)
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        xt = torch.from_numpy(x)
        t1 = time.perf_counter()
        out = self._method.execute((xt,))
        t2 = time.perf_counter()
        pred = out[0].detach().cpu().numpy() if hasattr(out[0], "detach") else np.asarray(out[0])
        h, w = bgr.shape[:2]
        dets = self.decode(pred, self.imgsz, r, px, py, w, h, self.class_names, self.conf_thr, self.iou_thr)
        t3 = time.perf_counter()
        return VisionResult(dets, severity_score(dets), (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, self.backend)


# --------------------------------------------------------------------------------------
# Deterministic rule engine (safety decision never waits for the LLM)
# --------------------------------------------------------------------------------------
def decide_disposition(spikes: int, severity: float, n_det: int, cfg: Config) -> Tuple[str, List[str]]:
    why: List[str] = []
    if n_det and severity >= cfg.stop_severity:
        why.append(f"defect severity {severity:.2f} >= {cfg.stop_severity:.2f}")
        return "STOP_LINE", why
    if n_det and spikes >= cfg.stop_spikes:
        why.append(f"defect detected AND machine burst ({spikes} >= {cfg.stop_spikes} spikes/ms)")
        return "STOP_LINE", why
    if n_det:
        why.append(f"{n_det} defect(s) detected, severity {severity:.2f}")
        return "QUARANTINE", why
    if spikes >= cfg.stop_spikes:
        why.append(f"no visual defect but strong machine anomaly ({spikes} spikes/ms)")
        return "QUARANTINE", why
    if spikes >= cfg.irq_spikes:
        why.append(f"no visual defect; hw_irq raised ({spikes} spikes/ms) -> watch")
        return "PASS", why
    why.append("no defect detected; machine signature nominal")
    return "PASS", why


# --------------------------------------------------------------------------------------
# Stage 2 — Llama 3.2 1B advisor via the ExecuTorch Llama runner
# --------------------------------------------------------------------------------------
@dataclass
class AdvisorResult:
    available: bool
    ok: bool = False                      # runner executed AND output parsed
    root_cause: str = ""
    corrective_action: str = ""
    raw_text: str = ""
    latency_ms: float = 0.0
    gen_tokens: int = 0
    tokens_estimated: bool = False
    tok_per_s: float = 0.0
    error: str = ""
    backend: str = ""


def load_kb(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_prompt(pkt: FPGASpikePacket, vis: VisionResult, disposition: str, kb: Dict[str, Any]) -> str:
    if vis.detections:
        d = vis.detections[0]
        entry = kb["classes"].get(d.class_name, {})
        hints = entry.get("sensor_hints", {})
        hint_txt = "; ".join(hints.get(s.lower(), "") for s in pkt.trigger_sources if s != "NONE" and hints.get(s.lower()))
        notes = (f"{entry.get('description', '')} Typical causes: {', '.join(entry.get('typical_causes', []))}. "
                 f"Standard actions: {', '.join(entry.get('corrective_actions', []))}. Sensor hints: {hint_txt or 'none'}.")
        defect_line = (f"DEFECT: {d.class_name} (confidence {d.conf:.2f}, severity {vis.severity:.2f}, "
                       f"covers {d.area_frac * 100:.0f}% of frame)")
    else:
        notes = " ".join(kb.get("machine_only", {}).get("corrective_actions", []))
        defect_line = "DEFECT: none detected"
    system = ("You are an offline quality-control advisor for a steel rolling line. The line disposition was already "
              "decided by a rule engine; do NOT change it. Use only the reference notes. "
              "Answer in exactly two lines:\nROOT_CAUSE: <one sentence>\nACTION: <one sentence>")
    user = (f"DISPOSITION: {disposition}\n{defect_line}\n"
            f"MACHINE SENSORS (1 ms window): spikes={pkt.spike_count} of max {SPIKES_PER_WINDOW_MAX}, "
            f"peak_vm={pkt.peak_vm:.1f} (threshold {V_THRESHOLD_AU:.0f}), channels={'+'.join(pkt.trigger_sources)}\n"
            f"REFERENCE NOTES: {notes}")
    # Llama-3 chat template. The runner adds BOS itself, so <|begin_of_text|> is intentionally omitted.
    return (f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n")


def parse_llm_text(text: str) -> Tuple[str, str]:
    rc = re.search(r"ROOT[_ ]CAUSE\s*:\s*(.+)", text, re.I)
    ac = re.search(r"ACTION\s*:\s*(.+)", text, re.I)
    return (rc.group(1).strip() if rc else ""), (ac.group(1).strip() if ac else "")


class LlamaAdvisor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def availability(self) -> Tuple[bool, str]:
        import shutil
        runner = shlex.split(self.cfg.llama_runner)
        if not runner or (shutil.which(runner[0]) is None and not Path(runner[0]).exists()):
            return False, f"Llama runner '{self.cfg.llama_runner}' not found (build ExecuTorch llama_main, set LLAMA_RUNNER)"
        for p, label in ((self.cfg.llama_pte, "model .pte"), (self.cfg.llama_tokenizer, "tokenizer")):
            if not Path(p).exists():
                return False, f"Llama {label} not found: {p}"
        return True, "ok"

    def _command(self, prompt: str) -> List[str]:
        subs = {"{model}": str(self.cfg.llama_pte), "{tokenizer}": str(self.cfg.llama_tokenizer),
                "{prompt}": prompt, "{seq_len}": str(self.cfg.llama_seq_len)}
        args = []
        for tok in shlex.split(self.cfg.llama_args):
            for k, v in subs.items():
                tok = tok.replace(k, v)
            args.append(tok)
        return shlex.split(self.cfg.llama_runner) + args

    @staticmethod
    def parse_stats(stdout: str, text: str, wall_s: float) -> Tuple[int, bool, float]:
        """-> (generated_tokens, estimated?, decode tokens/s). Uses the runner's PyTorchObserver stats when present."""
        m = re.search(r"PyTorchObserver\s+(\{.*\})", stdout)
        if m:
            try:
                s = json.loads(m.group(1))
                n = int(s.get("generated_tokens", 0))
                unit = float(s.get("SCALING_FACTOR_UNITS_PER_SECOND", 1000))
                t = (float(s["inference_end_ms"]) - float(s["prompt_eval_end_ms"])) / unit
                if n > 0 and t > 0:
                    return n, False, n / t
                if n > 0:
                    return n, False, n / max(wall_s, 1e-6)
            except (ValueError, KeyError, TypeError):
                pass
        n = max(1, int(len(text) / 4))                     # ~4 chars/token — flagged as an ESTIMATE
        return n, True, n / max(wall_s, 1e-6)

    def advise(self, pkt: FPGASpikePacket, vis: VisionResult, disposition: str, kb: Dict[str, Any]) -> AdvisorResult:
        ok, msg = self.availability()
        if not ok:
            return AdvisorResult(available=False, error=msg, backend="UNAVAILABLE")
        prompt = build_prompt(pkt, vis, disposition, kb)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(self._command(prompt), capture_output=True, text=True,
                                  timeout=self.cfg.llama_timeout_s)
        except subprocess.TimeoutExpired:
            return AdvisorResult(available=True, error=f"runner timed out after {self.cfg.llama_timeout_s}s",
                                 backend=self.cfg.llama_pte.name)
        wall = time.perf_counter() - t0
        out = proc.stdout
        if proc.returncode != 0:
            return AdvisorResult(available=True, error=f"runner exit {proc.returncode}: {proc.stderr[-300:]}",
                                 backend=self.cfg.llama_pte.name, latency_ms=wall * 1e3)
        text = out.split(prompt)[-1] if prompt in out else out          # drop prompt echo if the runner prints it
        text = re.sub(r"PyTorchObserver.*", "", text)
        text = re.split(r"<\|eot_id\|>|<\|end_of_text\|>", text)[0].strip()
        rc, ac = parse_llm_text(text)
        n, est, tps = self.parse_stats(out, text, wall)
        return AdvisorResult(available=True, ok=bool(rc and ac), root_cause=rc, corrective_action=ac, raw_text=text,
                             latency_ms=wall * 1e3, gen_tokens=n, tokens_estimated=est, tok_per_s=tps,
                             backend=f"ExecuTorch {self.cfg.llama_pte.name} ({self.cfg.llama_pte.stat().st_size / 2**20:.0f} MB)",
                             error="" if (rc and ac) else "output did not follow ROOT_CAUSE/ACTION format (raw text kept)")


# --------------------------------------------------------------------------------------
# Tamper-evident persistent audit log (SQLite + SHA-256 hash chain)
# --------------------------------------------------------------------------------------
class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL, hash TEXT NOT NULL)""")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _canon(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def append(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = row[0] if row else self.GENESIS
            ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            body = self._canon(payload)
            h = hashlib.sha256((prev + ts + body).encode()).hexdigest()
            cur = c.execute("INSERT INTO audit (ts, payload, prev_hash, hash) VALUES (?,?,?,?)", (ts, body, prev, h))
            return {"id": cur.lastrowid, "ts": ts, "hash": h, "prev_hash": prev, **payload}

    def rows(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rs = c.execute("SELECT id, ts, payload, hash FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": i, "ts": ts, "hash": h[:12], **json.loads(p)} for i, ts, p, h in rs]

    def verify(self) -> Tuple[bool, Optional[int]]:
        prev = self.GENESIS
        with self._conn() as c:
            for i, ts, payload, ph, h in c.execute("SELECT id, ts, payload, prev_hash, hash FROM audit ORDER BY id"):
                if ph != prev or hashlib.sha256((prev + ts + payload).encode()).hexdigest() != h:
                    return False, i
                prev = h
        return True, None

    def export_csv(self, path: Path) -> None:
        rows = self.rows(10**9)
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader(); w.writerows(reversed(rows))


# --------------------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------------------
def annotate(bgr: np.ndarray, dets: Sequence[Detection]) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = int(d.box[0] * w), int(d.box[1] * h), int(d.box[2] * w), int(d.box[3] * h)
        col = (0, 0, 255) if CLASS_CRITICALITY.get(d.class_name, 0.6) >= 0.75 else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{d.class_name} {d.conf:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


@dataclass
class InspectionResult:
    packet: FPGASpikePacket
    vision: VisionResult
    disposition: str
    reasons: List[str]
    advisor: Optional[AdvisorResult]
    kb_reference: List[str]
    annotated: np.ndarray
    record: Dict[str, Any]


class SpikeGuardPipeline:
    def __init__(self, cfg: Config, receiver: GowinSNNReceiver, vision: ExecuTorchVisionStage,
                 advisor: LlamaAdvisor, log: AuditLog):
        self.cfg, self.rx, self.vision, self.advisor, self.log = cfg, receiver, vision, advisor, log
        self.kb = load_kb(cfg.kb_path)

    def inspect(self, bgr: np.ndarray, image_name: str, scenario: Optional[str] = None,
                run_llm: bool = True) -> InspectionResult:
        pkt = self.rx.poll(scenario)
        vis = self.vision.infer(bgr)
        disp, why = decide_disposition(pkt.spike_count, vis.severity, len(vis.detections), self.cfg)
        adv = self.advisor.advise(pkt, vis, disp, self.kb) if run_llm else None
        kb_ref = (self.kb["classes"].get(vis.detections[0].class_name, {}).get("corrective_actions", [])
                  if vis.detections else self.kb.get("machine_only", {}).get("corrective_actions", []))
        ann = annotate(bgr, vis.detections)

        ok, buf = cv2.imencode(".png", bgr)
        sha = hashlib.sha256(buf.tobytes()).hexdigest() if ok else ""
        self.cfg.image_dir.mkdir(parents=True, exist_ok=True)
        img_path = self.cfg.image_dir / f"{int(time.time() * 1000)}_{sha[:8]}.png"
        cv2.imwrite(str(img_path), ann)

        top = vis.detections[0] if vis.detections else None
        payload = {
            "disposition": disp, "reasons": why,
            "defect": top.class_name if top else "none", "conf": round(top.conf, 3) if top else 0.0,
            "n_defects": len(vis.detections), "severity": round(vis.severity, 3),
            "spikes": pkt.spike_count, "peak_vm_au": round(pkt.peak_vm, 2), "channels": "+".join(pkt.trigger_sources),
            "stage0_source": pkt.source,
            "vision_ms": round(vis.total_ms, 2), "vision_infer_ms": round(vis.infer_ms, 2),
            "llm_ms": round(adv.latency_ms, 1) if adv else None,
            "llm_tok_per_s": round(adv.tok_per_s, 2) if adv and adv.ok else None,
            "llm_tokens_estimated": bool(adv.tokens_estimated) if adv else None,
            "llm_status": ("n/a" if adv is None else "ok" if adv.ok else adv.error or "unavailable"),
            "root_cause": adv.root_cause if adv and adv.ok else "",
            "action": adv.corrective_action if adv and adv.ok else "",
            "kb_reference_action": "; ".join(kb_ref[:2]),
            "image": image_name, "image_sha256": sha, "annotated_image": str(img_path),
            "vision_model": self.vision.backend, "llm_model": adv.backend if adv else "not run",
            "ruleset": f"stop_sev>={self.cfg.stop_severity},stop_spikes>={self.cfg.stop_spikes},irq>={self.cfg.irq_spikes}",
        }
        rec = self.log.append(payload)
        return InspectionResult(pkt, vis, disp, why, adv, kb_ref, ann, rec)


def build_pipeline(cfg: Optional[Config] = None, mode: str = "replay", port: str = "/dev/serial0",
                   baud: int = 460800) -> SpikeGuardPipeline:
    cfg = cfg or Config()
    rx = GowinSNNReceiver(mode=mode, replay_csv=cfg.replay_csv, port=port, baud=baud)
    vision = ExecuTorchVisionStage(cfg.vision_pte, cfg.imgsz, cfg.conf_thr, cfg.iou_thr)
    return SpikeGuardPipeline(cfg, rx, vision, LlamaAdvisor(cfg), AuditLog(cfg.db_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="SpikeGuard-QC single-shot inspection (CLI)")
    ap.add_argument("--image", required=True, help="path to a NEU-DET (or any) image")
    ap.add_argument("--scenario", default="vibration", help="replay scenario (see sim/uart_frames.csv)")
    ap.add_argument("--serial", default=None, help="serial port for a live FPGA, e.g. /dev/serial0 (overrides replay)")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    pipe = build_pipeline(mode="serial" if a.serial else "replay", port=a.serial or "/dev/serial0")
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read image {a.image}")
    r = pipe.inspect(img, Path(a.image).name, scenario=None if a.serial else a.scenario, run_llm=not a.no_llm)
    print(json.dumps({k: v for k, v in r.record.items() if k not in ("reasons",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
