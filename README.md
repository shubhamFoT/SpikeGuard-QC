# SpikeGuard-QC: 3-Stage Event-Driven Industrial Defect Detection & Offline QC Assistant

**Arm Bharat AI-SoC Challenge (2nd Edition) — Problem Statement 5**  
**Author:** Shubham  (`shubhamFoT`) | B.Tech ECE, Faculty of Technology, University of Delhi

---

## 1. System Architecture (3-Stage Edge Pipeline)

SpikeGuard-QC combines a sub-millisecond FPGA spiking anomaly trigger with quantized ExecuTorch models running on Arm Cortex-A NEON and Arm KleidiAI microkernels for offline industrial quality control and root-cause reporting.

```text
[Sensors: Acoustic / Vibration / Current]
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ Stage 0: Gowin GW1NR-9 FPGA (Verilog RTL)              │
│  • 27 MHz clock | 100 kHz LIF sample tick | 1 ms window │
│  • Q8.8 Leaky Integrate-and-Fire (LIF) neuron + IRQ    │
│  • 12-byte binary UART packetizer (115200 8N1, XOR CRC) │
└───────────────────────────┬────────────────────────────┘
                            │ UART (115200 8N1) + hw_irq
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Visual Defect Detection (ExecuTorch .pte)     │
│  • YOLO11n trained on NEU-DET (6 surface defect classes)│
│  • Input: (1, 3, 256, 256) BCHW → Output: (1, 10, 1344)│
│  • Target <25 ms/frame on Arm Cortex-A NEON (XNNPACK)  │
└───────────────────────────┬────────────────────────────┘
                            │ Fused [FPGA Spikes + Vision Defect]
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Offline QC Assistant (ExecuTorch .pte)        │
│  • Llama-3.2-1B-Instruct quantized to 8da4w + 8w        │
│  • Accelerated via XNNPACK / Arm KleidiAI microkernels  │
│  • Outputs PASS / QUARANTINE / STOP-LINE + root cause │
│    and corrective action with ISO-9001 / IPC traceability│
└────────────────────────────────────────────────────────┘

SpikeGuard-QC/
├── fpga/
│   ├── lif_neuron_core.v            # Q8.8 LIF neuron core
│   ├── snn_uart_packetizer.v        # 12-byte UART packetizer (115200 8N1)
│   ├── spikeguard_top.v             # Top-level 27 MHz / 100 kHz / 1 ms SNN + IRQ + UART
│   └── gowin_gw1nr9_spikeguard.cst  # Tang Nano 9K (GW1NR-9) pin constraints
├── .gitignore
├── benchmark_track_a_d.py           # .pte verification & latency benchmark runner
├── edge_pipeline_runner.py          # 3-stage pipeline runner (SNN + Vision + QC LLM)
├── shopfloor_dashboard.py           # Streamlit shop-floor operator dashboard
└── README.md