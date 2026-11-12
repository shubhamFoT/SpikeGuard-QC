"""Run the digital twin as a Modbus server.

  python -m spikeguard_twin.run_twin --schedule healthy:20,bpfo_mild:15,bpfo_severe:20,sensor_fault:10
  python -m spikeguard_twin.run_twin --cwru data/cwru/97.mat --cwru-fault data/cwru/105.mat

Segments: healthy | bpfo_mild | bpfo_severe | bpfi_severe | sensor_fault   (seconds after the colon)."""
import argparse
import itertools
import time

from config import CFG
from common.schemas import MachineState
from spikeguard_twin.modbus_tcp import ModbusServer
from spikeguard_twin.rf_snn import SpikeGuardTwin, estimate_shaft_freq
from spikeguard_twin.signals import synth_vibration, load_cwru, windows

C = CFG.twin
SEGMENTS = {"healthy": (None, 0.0), "bpfo_mild": ("BPFO", 0.6), "bpfo_severe": ("BPFO", 1.5),
            "bpfi_severe": ("BPFI", 1.0)}


def build_twin(rpm=1797.0, seed=0):
    healthy = [synth_vibration(C.fs, C.window_s, rpm, None, seed=seed + i)[0] for i in range(8)]
    return SpikeGuardTwin(C.fs).calibrate(healthy, rpm / 60.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5020)
    ap.add_argument("--schedule", default="healthy:30")
    ap.add_argument("--loop", action="store_true", help="repeat the schedule forever")
    ap.add_argument("--cwru", help="CWRU healthy .mat (calibration + healthy segments)")
    ap.add_argument("--cwru-fault", help="CWRU faulty .mat (used for fault segments)")
    a = ap.parse_args()

    server = ModbusServer(a.host, a.port).start()
    rpm = 1797.0
    cw_h = cw_f = None
    if a.cwru:
        sig, r = load_cwru(a.cwru)
        rpm = r or rpm
        cw_h = list(windows(sig, C.fs, C.window_s))
        twin = SpikeGuardTwin(C.fs).calibrate(cw_h[:8], rpm / 60.0)
        cw_h = itertools.cycle(cw_h[8:] or cw_h)
        if a.cwru_fault:
            cw_f = itertools.cycle(list(windows(load_cwru(a.cwru_fault)[0], C.fs, C.window_s)))
    else:
        twin = build_twin(rpm)
    print(f"twin ready on {a.host}:{a.port}  (rpm {rpm}, shaft {rpm/60:.1f} Hz)")

    plan = [(s.split(":")[0], float(s.split(":")[1])) for s in a.schedule.split(",")]
    seed = 1000
    while True:
        for name, secs in plan:
            t_end = time.time() + secs
            while time.time() < t_end:
                seed += 1
                if name == "sensor_fault":
                    st = MachineState(time.time(), 3.5, "fault", 0.0, 0.0)
                elif cw_h is not None:
                    w = next(cw_f) if (name != "healthy" and cw_f) else next(cw_h)
                    st = twin.process(w, estimate_shaft_freq(w, C.fs), time.time())
                else:
                    fault, sev = SEGMENTS[name]
                    w, _ = synth_vibration(C.fs, C.window_s, rpm, fault, sev, seed=seed)
                    st = twin.process(w, ts=time.time())
                server.bank.publish(st)
                print(f"[{name:12s}] {st.loop_ma:5.2f} mA  {st.zone:8s} rate {st.spike_rate_hz:6.1f} Hz")
                time.sleep(C.window_s)
        if not a.loop:
            break


if __name__ == "__main__":
    main()
