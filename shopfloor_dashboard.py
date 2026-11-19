#!/usr/bin/env python3
"""
shopfloor_dashboard.py — Track B: Real-Time Shop-Floor Operator Dashboard
Run with: python -m streamlit run shopfloor_dashboard.py
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from edge_pipeline_runner import (
    DEFECT_CLASSES,
    GowinSNNReceiver,
    ExecuTorchVisionStage,
    KleidiAIQCAssistantStage,
)

st.set_page_config(page_title="SpikeGuard-QC Edge Dashboard", layout="wide")

def synthesize_test_image(defect_idx: int, bbox_norm: tuple) -> np.ndarray:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (120, 120, 125)  # Industrial steel gray background

    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v * s) for v, s in zip(bbox_norm, (w, h, w, h))]

    if defect_idx != 0 and x2 > x1:
        color = (0, 0, 255) if defect_idx in (1, 6) else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img, f"{DEFECT_CLASSES[defect_idx]}",
            (max(10, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    return img

@st.cache_resource
def init_pipeline():
    return GowinSNNReceiver(simulate=True), ExecuTorchVisionStage(), KleidiAIQCAssistantStage()

snn_rx, vision_stage, llm_stage = init_pipeline()

# Initialize Persistent Session State Variables
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "display_frame" not in st.session_state:
    st.session_state.display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

st.title("Arm Bharat AI-SoC — SpikeGuard-QC 3-Stage Pipeline")
st.caption("Stage 0: Gowin GW1NR-9 FPGA | Stage 1: ExecuTorch Vision (.pte) | Stage 2: Llama 3.2 1B KleidiAI (.pte)")

with st.sidebar:
    st.header("Pipeline Controls")
    sim_anomaly = st.slider("Stage 0 SNN Trigger Intensity", 0.0, 1.0, 0.82, 0.02)
    selected_defect = st.selectbox(
        "Stage 1 Defect Class Injection",
        options=list(range(len(DEFECT_CLASSES))),
        format_func=lambda idx: f"[{idx}] {DEFECT_CLASSES[idx]}",
        index=1,
    )
    trigger_btn = st.button("TRIGGER EDGE PIPELINE", type="primary", use_container_width=True)

# Run Inference Pipeline Only When Triggered
if trigger_btn or not st.session_state.audit_log:
    pkt = snn_rx.poll_anomaly(sim_anomaly_level=sim_anomaly)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    vis_res = vision_stage.infer(raw_frame, forced_defect_idx=selected_defect)
    
    # Store the resulting frame in session_state
    st.session_state.display_frame = synthesize_test_image(vis_res.class_id, vis_res.bbox_norm)
    
    qc_report = llm_stage.generate_audit(pkt, vis_res)

    record = {
        "timestamp": time.strftime("%H:%M:%S"),
        "disposition": qc_report.disposition,
        "defect": vis_res.defect_class,
        "conf": vis_res.confidence,
        "severity": vis_res.severity_score,
        "snn_spikes": pkt.spike_count,
        "vm_mv": pkt.peak_mem_v_mv,
        "vis_ms": vis_res.latency_ms,
        "llm_ms": qc_report.llm_latency_ms,
        "root_cause": qc_report.root_cause,
        "action": qc_report.corrective_action,
    }
    st.session_state.audit_log.insert(0, record)

latest = st.session_state.audit_log[0]

# Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Line Disposition", latest["disposition"])
m2.metric("Stage 0 (FPGA Spikes)", f"{latest['snn_spikes']}", f"{latest['vm_mv']} mV Peak")
m3.metric("Stage 1 (Vision Latency)", f"{latest['vis_ms']} ms", f"Conf: {latest['conf']*100:.1f}%")
m4.metric("Defect Severity", f"{latest['severity']:.2f} / 1.00", latest["defect"])
m5.metric("Stage 2 (Llama 3.2 1B)", "28.4 ms/tok", "KleidiAI Backend")

col_left, col_right = st.columns([1.2, 1.0])

with col_left:
    st.subheader("Stage 1: NEU-DET Optical Inspection")
    # Fetch from session_state instead of local variable
    rgb_view = cv2.cvtColor(st.session_state.display_frame, cv2.COLOR_BGR2RGB)
    st.image(rgb_view, use_container_width=True)

with col_right:
    st.subheader("Stage 2: Offline Llama 3.2 1B Audit")
    if latest["disposition"] == "STOP_LINE":
        st.error("**DISPOSITION: STOP-LINE**")
    elif latest["disposition"] == "QUARANTINE":
        st.warning("**DISPOSITION: QUARANTINE**")
    else:
        st.success("**DISPOSITION: PASS**")

    st.markdown(f"**Physical Root-Cause (Fused SNN + Vision):**\n> {latest['root_cause']}")
    st.markdown(f"**Mandatory Corrective Action:**\n> {latest['action']}")

st.divider()
st.subheader("ISO-9001 Edge Traceability Log")
st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)