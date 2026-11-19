#!/usr/bin/env python3
"""
edge_pipeline_runner.py — Track B: Unified 3-Stage Edge Inference Runner
Integrates:
  - Stage 0: Gowin GW1NR-9 LIF SNN UART Binary Frame Parser
  - Stage 1: ExecuTorch YOLO11n NEU-DET Vision (.pte)
  - Stage 2: Llama 3.2 1B-Instruct QC Assistant (.pte)
"""

import struct
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ExecuTorch is optional on Windows for dashboard simulation
try:
    from executorch.runtime import Runtime
    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False

DEFECT_CLASSES = [
    "NOMINAL_PASS",
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IPC_STANDARD_REFS = {
    "NOMINAL_PASS": "ISO-9001 Nominal Profile",
    "crazing": "IPC-A-610H Sec 10.2.1 (Surface Crazing - Class 3)",
    "inclusion": "IPC-A-610H Sec 10.2.2 (Foreign Material Inclusion)",
    "patches": "IPC-A-610H Sec 10.2.3 (Plating Patches / Oxidation)",
    "pitted_surface": "IPC-A-610H Sec 10.2.4 (Surface Pitting)",
    "rolled-in_scale": "IPC-A-610H Sec 10.2.5 (Rolled-in Scale / Slag)",
    "scratches": "IPC-A-610H Sec 10.2.6 (Substrate Scratches)",
}

@dataclass
class FPGASpikePacket:
    timestamp: float
    trig_flags: int          
    spike_count: int         
    peak_mem_v_mv: float     
    delta_rate_hz: int       
    window_us: int           
    valid_crc: bool

    @property
    def trigger_sources(self) -> list:
        sources = []
        if self.trig_flags & 0x01: sources.append("ACOUSTIC_EMISSION")
        if self.trig_flags & 0x02: sources.append("VIBRATION_HARMONIC")
        if self.trig_flags & 0x04: sources.append("SPINDLE_CURRENT_SPIKE")
        return sources or ["PERIODIC_HEARTBEAT"]

@dataclass
class VisionInferenceResult:
    defect_class: str
    class_id: int
    confidence: float
    bbox_norm: Tuple[float, float, float, float]
    severity_score: float
    latency_ms: float
    backend: str

@dataclass
class QCAuditReport:
    disposition: str
    ipc_clause: str
    root_cause: str
    corrective_action: str
    iso9001_cap_code: str
    llm_latency_ms: float

class GowinSNNReceiver:
    def __init__(self, simulate: bool = True):
        self.simulate = simulate

    def poll_anomaly(self, sim_anomaly_level: float = 0.78) -> FPGASpikePacket:
        # Deterministic simulation of the 12-byte UART FPGA packet
        is_spike = sim_anomaly_level > 0.45
        flags = 0x07 if sim_anomaly_level > 0.8 else (0x03 if is_spike else 0x00)
        spikes = int(12 + sim_anomaly_level * 145)
        peak_q88 = int((0.45 + sim_anomaly_level * 0.95) * 256)
        delta_hz = int(800 + sim_anomaly_level * 14200)
        
        return FPGASpikePacket(
            timestamp=time.time(),
            trig_flags=flags,
            spike_count=spikes,
            peak_mem_v_mv=round(peak_q88 / 256.0, 2),
            delta_rate_hz=delta_hz,
            window_us=1000,
            valid_crc=True,
        )

class ExecuTorchVisionStage:
    def __init__(self, pte_path: str = "runs/neu_y11n/weights/best_executorch_model/best.pte"):
        self.pte_path = Path(pte_path)
        self.backend_label = "ExecuTorch XNNPACK INT8 (Simulated)"

    def infer(self, bgr_frame: np.ndarray, forced_defect_idx: int = 1) -> VisionInferenceResult:
        t0 = time.perf_counter()
        
        # Simulate edge execution latency for the Streamlit dashboard
        time.sleep(0.069) 
        dt_ms = (time.perf_counter() - t0) * 1000.0
        
        cls_id = forced_defect_idx
        confidence = 0.942 if cls_id != 0 else 0.985
        bbox = (0.31, 0.27, 0.64, 0.69) if cls_id != 0 else (0.0, 0.0, 0.0, 0.0)
        severity = 0.86 if cls_id in (1, 2, 5) else (0.58 if cls_id in (3, 4) else 0.04)

        return VisionInferenceResult(
            defect_class=DEFECT_CLASSES[cls_id],
            class_id=cls_id,
            confidence=round(confidence, 4),
            bbox_norm=bbox,
            severity_score=round(severity, 3),
            latency_ms=round(dt_ms, 2),
            backend=self.backend_label,
        )

class KleidiAIQCAssistantStage:
    def __init__(self, llama_pte_path: str = "models/llama_et/model.pte"):
        self.llama_pte_path = Path(llama_pte_path)

    @staticmethod
    def determine_disposition(fpga: FPGASpikePacket, vision: VisionInferenceResult) -> str:
        if vision.class_id == 0 and fpga.spike_count < 40:
            return "PASS"
        if vision.severity_score >= 0.78 or (fpga.spike_count > 110 and vision.class_id != 0):
            return "STOP_LINE"
        return "QUARANTINE"

    def generate_audit(self, fpga: FPGASpikePacket, vision: VisionInferenceResult) -> QCAuditReport:
        disposition = self.determine_disposition(fpga, vision)
        ipc_clause = IPC_STANDARD_REFS.get(vision.defect_class, "IPC-A-610H General Class 3")
        
        t0 = time.perf_counter()
        time.sleep(0.085)  # Simulated LLM generation delay for Windows demo
        dt_ms = (time.perf_counter() - t0) * 1000.0

        root_causes = {
            "NOMINAL_PASS": "Acoustic/vibration signature and surface geometry within nominal IPC envelope.",
            "crazing": f"Thermal shock during cooldown causing micro-fractures (SNN spikes={fpga.spike_count} at {fpga.delta_rate_hz} Hz).",
            "inclusion": f"Foreign debris trapped during rolling phase; detected via acoustic transient (Vm={fpga.peak_mem_v_mv} mV).",
            "patches": "Uneven chemical bath concentration leading to localized oxidation patches.",
            "pitted_surface": "Gas entrapment during casting causing micro-pitting on substrate.",
            "rolled-in_scale": "Incomplete descaling before rolling mill engagement.",
            "scratches": f"High-frequency abrasion from misaligned guide rollers ({fpga.delta_rate_hz} Hz delta-mod spike)."
        }

        actions = {
            "NOMINAL_PASS": "Continue conveyor indexing; log heartbeat frame.",
            "crazing": "Halt line; recalibrate cooling zone gradient to <= 2.5 C/s.",
            "inclusion": "Divert to quarantine; inspect air filtration system above rolling line.",
            "patches": "Flush and balance chemical bath pH; check agitation pumps.",
            "pitted_surface": "Review casting degassing protocol and temperature.",
            "rolled-in_scale": "Increase descaler water pressure by 15 PSI.",
            "scratches": "Immediate STOP-LINE; inspect and realign guide rollers on Stand 4."
        }

        return QCAuditReport(
            disposition=disposition,
            ipc_clause=ipc_clause,
            root_cause=root_causes.get(vision.defect_class, "Unknown anomaly."),
            corrective_action=actions.get(vision.defect_class, "Manual inspection required."),
            iso9001_cap_code=f"CAPA-2026-{vision.class_id:02d}X",
            llm_latency_ms=round(dt_ms, 1)
        )