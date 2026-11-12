# SpikeGuard-QC

Real-time industrial defect detection and QC assistant (Arm Bharat AI-SoC Challenge, PS5) that also
reads bearing-health from a SpikeGuard node (software twin in Phase 1, hardware later).

Vision runs on-device (YOLO, exported to ExecuTorch). A Llama 3.2 Instruct model, deployed with
ExecuTorch, writes the operator advisory. Pass/reject and fusion are deterministic code.

Run order (details in the build guide):
1. `python -m vision.prepare_neu --src <NEU-DET folder> --out data/neu_yolo`
2. `python -m vision.train`             -> runs/neu_y11n/weights/best.pt
3. `python -m vision.export_executorch --weights runs/neu_y11n/weights/best.pt`
4. `python -m spikeguard_twin.run_twin --schedule healthy:30,bpfo_mild:20,bpfo_severe:20,sensor_fault:10 --loop`
5. `uvicorn app.server:app --host 0.0.0.0 --port 8000`   (see app/server.py for environment variables)
6. `python -m scripts.benchmark_vision ...`, `python -m scripts.benchmark_llm ...`, `python -m scripts.eval_twin`, `python -m scripts.eval_fusion`
7. `python -m pytest -q`

Licence note: Ultralytics YOLO is AGPL-3.0; choose your repo licence accordingly.
