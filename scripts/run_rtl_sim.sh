#!/usr/bin/env bash
# Compile + run the Stage-0 RTL regression. Usage: scripts/run_rtl_sim.sh [vcd]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sim
iverilog -g2005 -Wall -o sim/tb_spikeguard \
    tb/tb_spikeguard_top.v fpga/lif_neuron_core.v fpga/snn_uart_packetizer.v fpga/spikeguard_top.v
if [[ "${1:-}" == "vcd" ]]; then vvp sim/tb_spikeguard +vcd | tee sim/rtl_regression.log
else vvp sim/tb_spikeguard | tee sim/rtl_regression.log; fi
grep -q "RESULT: PASS" sim/rtl_regression.log
echo "UART frames written to sim/uart_frames.csv"
