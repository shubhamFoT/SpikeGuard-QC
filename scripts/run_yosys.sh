#!/usr/bin/env bash
# Re-run synthesis and print the resource table (Yosys 0.33, synth_gowin)
set -euo pipefail
cd "$(dirname "$0")/.."
yosys -q -l sim/yosys.log -p "read_verilog fpga/lif_neuron_core.v fpga/snn_uart_packetizer.v fpga/spikeguard_top.v; synth_gowin -top spikeguard_top; stat" >/dev/null
sed -n '/Number of cells/,/^$/p' sim/yosys.log | tail -20
# Optional place-and-route / timing / bitstream (needs the OSS CAD Suite: nextpnr-gowin, gowin_pack):
#   yosys -p "read_verilog fpga/*.v; synth_gowin -top spikeguard_top -json sim/top.json"
#   nextpnr-gowin --json sim/top.json --write sim/pnr.json --device GW1NR-LV9QN88PC6/I5 \
#                 --family GW1N-9C --cst fpga/gowin_gw1nr9_spikeguard.cst --freq 27
#   gowin_pack -d GW1N-9C -o sim/spikeguard.fs sim/pnr.json
