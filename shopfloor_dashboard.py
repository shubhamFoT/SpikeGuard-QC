"""
shopfloor_dashboard.py — SpikeGuard-QC operator console (Streamlit)

Every figure on this page is measured or computed at run time:
  * Stage 0: real UART frames (live serial) or frames produced by the RTL simulation (labelled as such)
  * Stage 1: real ExecuTorch .pte inference on the real image shown
  * Stage 2: real Llama .pte via the ExecuTorch runner; if unavailable the page SAYS so
  * Log   : persistent SQLite, hash-chained (survives refresh/restart)

Run inside WSL2/Linux:   streamlit run shopfloor_dashboard.py
"""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import edge_pipeline_runner as epr

st.set_page_config(page_title="SpikeGuard-QC", layout="wide")
cfg = epr.Config()
# Streamlit >=1.50 deprecates use_container_width in favour of width="stretch"; support both.
_V = tuple(int(x) for x in (st.__version__.split(".") + ["0"])[:2])
FIT = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
BANNER = {"PASS": st.success, "QUARANTINE": st.warning, "STOP_LINE": st.error}


@st.cache_resource(show_spinner="Loading models…")
def get_pipeline(mode: str, port: str, baud: int):
    return epr.build_pipeline(cfg, mode=mode, port=port, baud=baud)


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Pipeline controls")
mode = st.sidebar.radio("Stage 0 source", ["RTL simulation replay", "Live FPGA (serial)"])
port = st.sidebar.text_input("Serial port", "/dev/serial0") if mode.startswith("Live") else "/dev/serial0"
baud = 460800
img_src = st.sidebar.radio("Image source", ["Image folder", "Upload", "Webcam"])
run_llm = st.sidebar.checkbox("Run Llama advisor (Stage 2)", value=True)

try:
    pipe = get_pipeline("serial" if mode.startswith("Live") else "replay", port, baud)
except Exception as e:                                   # noqa: BLE001 — show the real reason, never fake it
    st.title("SpikeGuard-QC")
    st.error(f"Pipeline could not start: {e}")
    st.info("Checklist: run inside WSL2/Linux · `pip install executorch torch` · best.pte present "
            "(SPIKEGUARD_VISION_PTE) · `scripts/run_rtl_sim.sh` executed (creates sim/uart_frames.csv).")
    st.stop()

scenario = None
if mode.startswith("RTL"):
    scenario = st.sidebar.selectbox("Machine scenario (RTL-simulated sensor input)", pipe.rx.scenarios(),
                                    index=min(2, len(pipe.rx.scenarios()) - 1))

image, image_name = None, ""
if img_src == "Image folder":
    folder = Path(st.sidebar.text_input("Folder", str(epr.ROOT / "data/test_images")))
    files = sorted([p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]) if folder.exists() else []
    if not files:
        st.sidebar.warning("No images found. Copy NEU-DET *test* images into this folder.")
    else:
        pick = st.sidebar.selectbox("Image", [p.name for p in files])
        if st.sidebar.button("Random image"):
            pick = random.choice(files).name
        image_name = pick
        image = cv2.imread(str(folder / pick))
elif img_src == "Upload":
    up = st.sidebar.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp"])
    if up is not None:
        image_name = up.name
        image = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    cam = st.camera_input("Inspection camera")
    if cam is not None:
        image_name = "webcam_capture"
        image = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)

go = st.sidebar.button("RUN INSPECTION", type="primary", disabled=image is None)

# ------------------------------------------------------------------ run
st.title("Arm Bharat AI-SoC — SpikeGuard-QC 3-Stage Pipeline")
if go and image is not None:
    with st.spinner("Inspecting…"):
        st.session_state["last"] = pipe.inspect(image, image_name, scenario=scenario, run_llm=run_llm)

r = st.session_state.get("last")
if r is None:
    st.info("Pick an image and a machine scenario, then press RUN INSPECTION.")
else:
    adv, vis, pkt = r.advisor, r.vision, r.packet
    p0, p1, p2 = st.columns(3)
    p0.caption(f"**Stage 0** — {pkt.source}")
    p1.caption(f"**Stage 1** — {vis.backend}")
    p2.caption("**Stage 2** — " + (adv.backend if adv and adv.available else "UNAVAILABLE" if adv else "not run"))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Line disposition", r.disposition)
    k2.metric("Stage 0 spikes / 1 ms", f"{pkt.spike_count} / {epr.SPIKES_PER_WINDOW_MAX}",
              f"peak Vm {pkt.peak_vm:.1f} a.u. · {'+'.join(pkt.trigger_sources)}", delta_color="off")
    k3.metric("Stage 1 latency (measured)", f"{vis.infer_ms:.1f} ms",
              f"pre {vis.pre_ms:.1f} · post {vis.post_ms:.1f} ms", delta_color="off")
    k4.metric("Defect severity", f"{vis.severity:.2f} / 1.00",
              r.vision.detections[0].class_name if vis.detections else "no defect", delta_color="off")
    if adv and adv.ok:
        k5.metric("Stage 2 Llama (measured)", f"{adv.tok_per_s:.1f} tok/s",
                  f"{adv.gen_tokens} tok{' (est.)' if adv.tokens_estimated else ''} · {adv.latency_ms / 1000:.1f} s total",
                  delta_color="off")
    else:
        k5.metric("Stage 2 Llama", "—", "unavailable" if adv is None or not adv.available else "see message", delta_color="off")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Stage 1: NEU-DET optical inspection")
        st.image(cv2.cvtColor(r.annotated, cv2.COLOR_BGR2RGB), **FIT,
                 caption=f"{image_name} — boxes are real model output" if vis.detections else f"{image_name} — no defect detected")
        if vis.detections:
            st.dataframe(pd.DataFrame([{"class": d.class_name, "confidence": round(d.conf, 3),
                                        "area %": round(d.area_frac * 100, 1)} for d in vis.detections]),
                         hide_index=True, **FIT)
    with right:
        st.subheader("Rule engine + Stage 2 advisor")
        BANNER[r.disposition](f"DISPOSITION: {r.disposition}  —  " + "; ".join(r.reasons))
        if adv is None:
            st.caption("Llama advisor not run (disabled in sidebar).")
        elif adv.ok:
            st.markdown(f"**Physical root cause (Llama 3.2 1B, offline):** {adv.root_cause}")
            st.markdown(f"**Mandatory corrective action:** {adv.corrective_action}")
        else:
            st.warning(f"Llama advisor: {adv.error or 'unavailable'}")
            if adv.raw_text:
                st.code(adv.raw_text)
        with st.expander("Knowledge-base reference actions (retrieval, not LLM)"):
            for a in r.kb_reference:
                st.write("• " + a)

# ------------------------------------------------------------------ audit log
st.divider()
st.subheader("ISO 9001 edge traceability log (persistent, hash-chained)")
rows = pipe.log.rows(100)
if rows:
    cols = ["id", "ts", "disposition", "defect", "conf", "severity", "spikes", "peak_vm_au", "channels",
            "vision_ms", "llm_ms", "llm_tok_per_s", "llm_status", "root_cause", "action", "image", "hash"]
    st.dataframe(pd.DataFrame(rows).reindex(columns=cols), hide_index=True, **FIT)
c1, c2, _ = st.columns([1, 1, 4])
if c1.button("Verify log integrity"):
    ok, bad = pipe.log.verify()
    (st.success("Hash chain intact — no record has been altered.") if ok else st.error(f"Tampering detected at record id {bad}"))
if rows:
    tmp = cfg.db_path.with_suffix(".csv")
    pipe.log.export_csv(tmp)
    c2.download_button("Export CSV", tmp.read_bytes(), file_name="qc_audit_export.csv", mime="text/csv")
