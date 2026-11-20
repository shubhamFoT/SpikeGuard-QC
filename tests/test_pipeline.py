import shlex
"""
Unit tests for everything in edge_pipeline_runner.py that does not need the real .pte models.
Run:  pytest -q          (after scripts/run_rtl_sim.sh so sim/uart_frames.csv exists)
Test doubles (a fake Llama runner, a stub vision stage) live ONLY in this file.
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import edge_pipeline_runner as epr  # noqa: E402

CSV = ROOT / "sim" / "uart_frames.csv"


# ---------------------------------------------------------------- Stage 0
def make_frame(flags=0b010, spikes=20, peak=0x4600, delta=3, win=1000) -> bytes:
    body = bytes([flags, spikes & 255, spikes >> 8, peak & 255, peak >> 8, delta & 255, delta >> 8,
                  win & 255, win >> 8])
    return bytes([0xA5]) + body + bytes([epr.xor_crc(body), 0x5A])


def test_parse_valid_frame():
    p = epr.parse_frame(make_frame())
    assert p and p.spike_count == 20 and p.trig_flags == 0b010 and p.window_us == 1000
    assert p.trigger_sources == ["VIBRATION"] and p.hw_irq


def test_parse_rejects_bad_crc_and_markers():
    f = bytearray(make_frame()); f[10] ^= 0xFF
    assert epr.parse_frame(bytes(f)) is None
    f = bytearray(make_frame()); f[0] = 0x00
    assert epr.parse_frame(bytes(f)) is None
    assert epr.parse_frame(make_frame()[:-1]) is None


def test_parser_resyncs_after_garbage_and_corruption():
    good = make_frame(spikes=11)
    bad = bytearray(make_frame(spikes=50)); bad[5] ^= 0x01            # corrupted (CRC fails)
    stream = b"\x00\xA5\x13" + bytes(bad) + good + good
    parser = epr.FrameParser()
    pkts = parser.feed(stream[:7]) + parser.feed(stream[7:])           # split across reads
    assert [p.spike_count for p in pkts] == [11, 11]
    assert parser.bad_frames >= 1


@pytest.mark.skipif(not CSV.exists(), reason="run scripts/run_rtl_sim.sh first")
def test_replay_matches_rtl_physical_limits():
    rx = epr.GowinSNNReceiver("replay", CSV)
    assert rx.poll("idle").spike_count == 0
    assert rx.poll("all_channels").spike_count == epr.SPIKES_PER_WINDOW_MAX      # hardware ceiling = 50
    assert 18 <= rx.poll("vibration").spike_count <= 21 and rx.poll("vibration").hw_irq
    assert not rx.poll("acoustic").hw_irq
    g = rx.poll("glitch_10us")
    assert g.spike_count == 0 and g.peak_vm > 0 and not g.hw_irq                 # glitch seen, not fired
    assert all(rx.poll(s).spike_count <= epr.SPIKES_PER_WINDOW_MAX for s in rx.scenarios())


# ---------------------------------------------------------------- Stage 1
def fake_pred(dets, n=1344, nc=6):
    """dets: list of (cx,cy,w,h,cls,score) in 256-px letterbox space -> (1, 4+nc, n) like the .pte output."""
    p = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(dets):
        p[0, :4, i] = (cx, cy, w, h); p[0, 4 + c, i] = s
    return p


def test_decode_nms_and_class_awareness():
    pred = fake_pred([(128, 128, 100, 100, 2, 0.90),      # patches
                      (130, 128, 100, 100, 2, 0.80),      # duplicate of the same defect -> suppressed
                      (60, 60, 40, 40, 0, 0.70),          # crazing elsewhere
                      (200, 200, 20, 20, 5, 0.10)])       # below threshold
    d = epr.ExecuTorchVisionStage.decode(pred, 256, 1.0, 0, 0, 256, 256, epr.DEFECT_CLASSES, 0.25, 0.45)
    assert [(x.class_name, round(x.conf, 2)) for x in d] == [("patches", 0.9), ("crazing", 0.7)]
    assert np.allclose(d[0].box, (78 / 256, 78 / 256, 178 / 256, 178 / 256), atol=1e-3)


def test_letterbox_roundtrip_non_square():
    img = np.zeros((100, 200, 3), np.uint8)                # w=200,h=100 -> r=1.28, pad_y=64
    canvas, r, px, py = epr.ExecuTorchVisionStage.letterbox(img, 256)
    assert canvas.shape == (256, 256, 3) and abs(r - 1.28) < 1e-6 and (px, py) == (0, 64)
    pred = fake_pred([(128, 128, 64, 64, 1, 0.9)])         # centred box in letterbox space
    d = epr.ExecuTorchVisionStage.decode(pred, 256, r, px, py, 200, 100, epr.DEFECT_CLASSES, 0.25, 0.45)[0]
    assert np.allclose(d.box, (0.375, 0.25, 0.625, 0.75), atol=2e-2)      # back in ORIGINAL image coordinates


def test_decode_accepts_transposed_and_logit_outputs():
    pred = fake_pred([(128, 128, 50, 50, 3, 0.9)])
    d1 = epr.ExecuTorchVisionStage.decode(pred[0].T, 256, 1, 0, 0, 256, 256, epr.DEFECT_CLASSES, 0.25, 0.45)
    logit = pred.copy(); logit[0, 4:] = np.where(logit[0, 4:] > 0, 3.0, -8.0)
    d2 = epr.ExecuTorchVisionStage.decode(logit, 256, 1, 0, 0, 256, 256, epr.DEFECT_CLASSES, 0.25, 0.45)
    assert d1[0].class_name == d2[0].class_name == "pitted_surface"


def test_severity_monotonic_and_bounded():
    small = [epr.Detection(2, "patches", 0.6, (0.4, 0.4, 0.5, 0.5))]
    big = [epr.Detection(2, "patches", 0.6, (0.1, 0.1, 0.9, 0.9))]
    worse = [epr.Detection(1, "inclusion", 0.95, (0.1, 0.1, 0.9, 0.9))]
    assert epr.severity_score([]) == 0.0
    assert epr.severity_score(small) < epr.severity_score(big) < epr.severity_score(worse) <= 1.0
    assert epr.severity_score(worse + worse + worse) <= 1.0


# ---------------------------------------------------------------- rule engine
@pytest.mark.parametrize("spikes,sev,n,exp", [
    (0, 0.0, 0, "PASS"), (20, 0.0, 0, "PASS"), (40, 0.0, 0, "QUARANTINE"),
    (5, 0.4, 1, "QUARANTINE"), (35, 0.4, 1, "STOP_LINE"), (0, 0.80, 1, "STOP_LINE"),
])
def test_rules(spikes, sev, n, exp):
    assert epr.decide_disposition(spikes, sev, n, epr.Config())[0] == exp


# ---------------------------------------------------------------- Stage 2 (runner parsing, fake runner)
FAKE = textwrap.dedent('''
    import sys
    a = sys.argv
    prompt = a[a.index("--prompt") + 1]
    print(prompt + "ROOT_CAUSE: Scale not removed before the stand.\\nACTION: Raise descaler pressure.<|eot_id|>")
    print('PyTorchObserver {"generated_tokens":20,"prompt_eval_end_ms":1000,"inference_end_ms":3000,"SCALING_FACTOR_UNITS_PER_SECOND":1000}')
''')


def _cfg(tmp_path, runner):
    (tmp_path / "m.pte").write_bytes(b"x"); (tmp_path / "t.model").write_bytes(b"x")
    return epr.Config(llama_runner=runner, llama_pte=tmp_path / "m.pte", llama_tokenizer=tmp_path / "t.model",
                      db_path=tmp_path / "a.db", image_dir=tmp_path / "img")


def _pkt_vis():
    pkt = epr.parse_frame(make_frame())
    det = epr.Detection(4, "rolled-in_scale", 0.9, (0.2, 0.2, 0.6, 0.6))
    return pkt, epr.VisionResult([det], epr.severity_score([det]), 1, 2, 1, "stub")


def test_advisor_parses_text_and_runner_stats(tmp_path):
    (tmp_path / "fake.py").write_text(FAKE)
    cfg = _cfg(tmp_path, f"{shlex.quote(sys.executable)} {shlex.quote(str(tmp_path / 'fake.py'))}")
    pkt, vis = _pkt_vis()
    kb = epr.load_kb(cfg.kb_path)
    r = epr.LlamaAdvisor(cfg).advise(pkt, vis, "QUARANTINE", kb)
    assert r.available and r.ok and "descaler" in r.corrective_action and "Scale" in r.root_cause
    assert r.gen_tokens == 20 and not r.tokens_estimated and abs(r.tok_per_s - 10.0) < 1e-6


def test_advisor_reports_unavailable_instead_of_faking(tmp_path):
    cfg = _cfg(tmp_path, "definitely_not_a_real_binary")
    pkt, vis = _pkt_vis()
    r = epr.LlamaAdvisor(cfg).advise(pkt, vis, "PASS", epr.load_kb(cfg.kb_path))
    assert not r.available and not r.ok and "not found" in r.error and r.root_cause == ""


def test_prompt_contains_grounding_and_omits_bos():
    pkt, vis = _pkt_vis()
    p = epr.build_prompt(pkt, vis, "STOP_LINE", epr.load_kb(epr.Config().kb_path))
    assert "rolled-in_scale" in p and "descal" in p.lower() and "DISPOSITION: STOP_LINE" in p
    assert "<|begin_of_text|>" not in p and p.endswith("<|end_header_id|>\n\n")


# ---------------------------------------------------------------- audit log + full pipeline with a stub vision stage
def test_audit_chain_detects_tampering(tmp_path):
    log = epr.AuditLog(tmp_path / "a.db")
    for i in range(3):
        log.append({"n": i, "disposition": "PASS"})
    assert log.verify() == (True, None)
    import sqlite3
    with sqlite3.connect(tmp_path / "a.db") as c:
        c.execute("UPDATE audit SET payload = replace(payload, 'PASS', 'XXXX') WHERE id = 2")
    ok, bad = log.verify()
    assert not ok and bad == 2


class StubVision:                                     # test double — never used outside tests
    backend = "stub"
    def infer(self, bgr):
        d = epr.Detection(2, "patches", 0.7, (0.3, 0.3, 0.5, 0.5))      # moderate defect (severity ~0.2)
        return epr.VisionResult([d], epr.severity_score([d]), 1.0, 2.0, 1.0, "stub")


@pytest.mark.skipif(not CSV.exists(), reason="run scripts/run_rtl_sim.sh first")
def test_pipeline_end_to_end_persists_and_escalates(tmp_path):
    cfg = _cfg(tmp_path, "definitely_not_a_real_binary")
    pipe = epr.SpikeGuardPipeline(cfg, epr.GowinSNNReceiver("replay", CSV), StubVision(),
                                  epr.LlamaAdvisor(cfg), epr.AuditLog(cfg.db_path))
    img = np.full((200, 200, 3), 120, np.uint8)
    quiet = pipe.inspect(img, "a.png", scenario="idle")
    loud = pipe.inspect(img, "a.png", scenario="all_channels")
    assert quiet.packet.spike_count == 0 and quiet.disposition == "QUARANTINE"        # visual defect only
    assert loud.packet.spike_count == 50 and loud.disposition == "STOP_LINE"          # + machine burst -> escalates
    assert loud.record["llm_status"].startswith("Llama runner")                       # honest: LLM unavailable here
    assert loud.record["root_cause"] == "" and loud.record["kb_reference_action"]     # nothing fabricated
    # persistence: a brand-new AuditLog object sees both rows and the chain verifies
    again = epr.AuditLog(cfg.db_path)
    assert len(again.rows()) == 2 and again.verify() == (True, None)
