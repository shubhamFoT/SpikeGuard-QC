"""Run with:  python -m pytest -q     (or: python -m unittest discover)"""
import os
import tempfile
import time

import numpy as np

from app.db import DB
from app.pipeline import Pipeline
from common.schemas import MachineState
from fusion.rules import (AdvisoryGate, ISOLATED_DEFECT, MACHINE_WARNING_ONLY, MACHINE_WEAR, NORMAL,
                          HEALTH_UNAVAILABLE, PROCESS_OR_MATERIAL, STOP_AND_INSPECT, RollingRate, fuse)
from spikeguard_twin.modbus_tcp import MachineStateReader, ModbusServer
from spikeguard_twin.rf_snn import SpikeGuardTwin, csd_terms
from spikeguard_twin.signals import synth_vibration
from vision.metrics import binary_report, latency_stats
from vision.severity import Detection, grade, severity_from_area


def ms(zone, ma=4.0):
    return MachineState(time.time(), ma, zone, 0.0, 1.0)


def burst_rate():
    r = RollingRate()
    for _ in range(20):
        r.add(True)
    return r


# ---------- fusion table (section 4.3) ----------
def test_fusion_table():
    calm = RollingRate()
    for _ in range(20):
        calm.add(False)
    assert fuse(calm, False, ms("healthy")).category == NORMAL
    assert fuse(calm, True, ms("healthy")).category == ISOLATED_DEFECT
    assert fuse(burst_rate(), True, ms("healthy")).category == PROCESS_OR_MATERIAL
    assert fuse(burst_rate(), True, ms("warning")).category == MACHINE_WEAR
    assert fuse(calm, False, ms("warning")).category == MACHINE_WARNING_ONLY
    assert fuse(burst_rate(), True, ms("critical")).category == STOP_AND_INSPECT
    assert fuse(calm, False, ms("critical")).category == STOP_AND_INSPECT


def test_degradation_to_vision_only():
    f = fuse(burst_rate(), True, None)            # stale / unreachable node
    assert f.category == HEALTH_UNAVAILABLE and f.vision_only
    assert fuse(burst_rate(), True, ms("fault", 3.5)).category == HEALTH_UNAVAILABLE


def test_burst_needs_min_frames():
    r = RollingRate()
    for _ in range(3):
        r.add(True)                                # 100 % defective but only 3 frames
    assert not r.burst


def test_gate_cooldown_and_changes_only():
    g = AdvisoryGate(cooldown_s=10)
    assert not g.should_call(NORMAL, 0)
    assert g.should_call(MACHINE_WEAR, 1)
    assert not g.should_call(MACHINE_WEAR, 2)      # same category: no repeat
    assert not g.should_call(NORMAL, 3)
    assert not g.should_call(STOP_AND_INSPECT, 5)  # changed but inside cooldown
    g.should_call(NORMAL, 20)
    assert g.should_call(STOP_AND_INSPECT, 21)


# ---------- vision post-processing ----------
def test_severity_and_decision():
    assert severity_from_area(0) == "none"
    assert severity_from_area(0.01) == "minor"
    assert severity_from_area(0.05) == "major"
    assert severity_from_area(0.5) == "critical"
    big = Detection("scratches", 0.9, [0, 0, 100, 100])       # 25 % of a 200x200 frame
    assert grade([big], 200, 200)[4:] == ("critical", "reject")
    weak = Detection("crazing", 0.3, [0, 0, 10, 10])
    assert grade([weak], 200, 200)[5] == "review"
    assert grade([Detection("x", 0.1, [0, 0, 5, 5])], 200, 200)[5] == "pass"   # below conf_min


def test_metrics():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.4, 0.6, 0.2, 0.1, 0.1, 0.3]
    r = binary_report(y, s, 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 1, 4)
    assert abs(r["fpr"] - 0.2) < 1e-9 and 0 < r["auroc"] <= 1
    assert latency_stats([10, 20, 30])["median_ms"] == 20


# ---------- SpikeGuard twin ----------
def test_twin_separates_healthy_from_fault():
    fs = 12000
    tw = SpikeGuardTwin(fs).calibrate([synth_vibration(fs, 0.5, 1797, None, seed=s)[0] for s in range(8)],
                                      1797 / 60)
    healthy = [tw.process(synth_vibration(fs, 0.5, 1797, None, seed=300 + s)[0]).zone for s in range(10)]
    faulty = [tw.process(synth_vibration(fs, 0.5, 1797, "BPFO", 1.5, seed=300 + s)[0]).zone for s in range(10)]
    assert healthy.count("healthy") == 10
    assert faulty.count("critical") >= 9


def test_csd_terms_fewer_adders_for_round_numbers():
    assert csd_terms(0.5) == 1 and csd_terms(0.75) == 2


# ---------- Modbus ----------
def test_modbus_roundtrip_and_stale():
    srv = ModbusServer(port=5111).start()
    rd = MachineStateReader(port=5111, stale_after_s=0.4)
    assert rd.read() is None                                  # never published
    srv.bank.publish(ms("warning", 12.34))
    st = rd.read()
    assert st.zone == "warning" and abs(st.loop_ma - 12.34) < 0.01
    time.sleep(0.6)
    assert rd.read() is None                                  # heartbeat frozen -> stale
    srv.shutdown()


# ---------- whole pipeline with fakes ----------
class FakeDetector:
    def __init__(self):
        self.defective = False

    def infer(self, frame):
        d = [Detection("scratches", 0.9, [0, 0, 60, 60])] if self.defective else []
        return d, (200, 200)


class FakeSource:
    def next(self):
        return np.zeros((200, 200, 3), np.uint8), "x"


class FakeReader:
    def __init__(self):
        self.state = ms("healthy")

    def read(self):
        return self.state


class FakeAdvisor:
    def __init__(self):
        self.calls = []

    def advise(self, ctx):
        self.calls.append(ctx)
        return ({"summary": "s", "likely_cause": "c", "action": "a", "urgency": "medium"}, {"source": "fake"})


def test_pipeline_scenarios():
    tmp = tempfile.mkdtemp()
    det, rd, adv = FakeDetector(), FakeReader(), FakeAdvisor()
    p = Pipeline(det, FakeSource(), rd, adv, DB(os.path.join(tmp, "t.db")))
    p.gate.cooldown = 0
    for _ in range(12):                                      # scenario A: normal
        assert p.step()[1].category == NORMAL
    det.defective = True                                     # scenario B: burst, machine healthy
    cats = [p.step()[1].category for _ in range(30)]
    assert cats[-1] == PROCESS_OR_MATERIAL
    rd.state = ms("warning", 12)                             # scenario C: burst + bearing warning
    assert p.step()[1].category == MACHINE_WEAR
    rd.state = None                                          # scenario D: node lost
    assert p.step()[1].category == HEALTH_UNAVAILABLE
    while p.advise_once():                                   # drain advisory queue
        pass
    assert len(adv.calls) >= 1 and p.latest_advice is not None
    rows = p.db.recent("inspections", 100)
    assert len(rows) == 12 + 30 + 2
    assert p.db.export_csv("inspections", os.path.join(tmp, "o.csv")) == len(rows)
