# SpikeGuard-QC — Arm Bharat AI-SoC Challenge (PS5)
Event-aware industrial defect inspection: FPGA spiking front-end (Stage 0) → ExecuTorch YOLO11n (Stage 1) →
rule engine + offline Llama-3.2-1B advisor via ExecuTorch (Stage 2) → persistent hash-chained ISO-9001 audit log.

## What is real in this repo
| Stage | Phase-1 status | How to reproduce |
|---|---|---|
| 0 RTL | simulated (Icarus) + synthesised (Yosys 0.33) | `scripts/run_rtl_sim.sh`, `scripts/run_yosys.sh` |
| 1 Vision | real ExecuTorch .pte inference on real images | `python edge_pipeline_runner.py --image <img> --scenario vibration --no-llm` |
| 2 Llama | real if `docs/LLAMA_SETUP.md` completed; otherwise reported UNAVAILABLE (never faked) | `docs/LLAMA_SETUP.md` |
| Hardware | **not yet** (Phase 2): Tang Nano 9K + Raspberry Pi 5 | – |

## Quick start (WSL2 / Linux)
```bash
pip install -r requirements.txt torch executorch
bash scripts/run_rtl_sim.sh                 # makes sim/uart_frames.csv from the real RTL
pytest -q                                   # unit tests
streamlit run shopfloor_dashboard.py        # operator console
python benchmark_track_a_d.py --images data/test_images --llama-runs 3   # honest benchmark -> docs/benchmarks/
```
Place NEU-DET **test-split** images in `data/test_images/` and the exported model at
`runs/neu_y11n/weights/best_executorch_model/best.pte` (or set `SPIKEGUARD_VISION_PTE`).
Large files (.pte) are not in git; publish them as a GitHub Release and link here.

## Decision logic
`STOP_LINE` if severity ≥ 0.75, or defect ∧ ≥30 spikes/ms · `QUARANTINE` if any defect, or ≥30 spikes/ms ·
else `PASS`. Decided by a deterministic rule engine; Llama only explains (root cause + corrective action).
Thresholds: `Config` in `edge_pipeline_runner.py`. Stage-0 hardware limit: 50 spikes per 1 ms window.
