"""The real-time loop: frame -> detect -> grade -> fuse -> log, with Llama running OFF the hot path."""
import queue
import threading
import time

from config import CFG
from common.schemas import InspectionEvent
from fusion.rules import AdvisoryGate, RollingRate, fuse
from vision.severity import grade


class Pipeline:
    def __init__(self, detector, source, state_reader, advisor, db):
        self.detector, self.source, self.state_reader = detector, source, state_reader
        self.advisor, self.db = advisor, db
        self.rolling, self.gate = RollingRate(), AdvisoryGate()
        self.frame_id = 0
        self.q = queue.Queue(maxsize=1)           # at most one pending advisory request
        self.latest_advice = None
        self.latest_status = {}
        self.latest_frame = None
        self._stop = threading.Event()

    # --- one iteration; synchronous so it is easy to test ---------------------------------
    def step(self):
        frame, truth = self.source.next()
        t0 = time.perf_counter()
        dets, (w, h) = self.detector.infer(frame)
        cls, conf, bbox, area, sev, decision = grade(dets, w, h)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        defective = decision == "reject"
        self.rolling.add(defective)
        machine = self.state_reader.read()                    # None if stale or unreachable
        fused = fuse(self.rolling, defective, machine)

        now = time.time()
        self.frame_id += 1
        ev = InspectionEvent(now, self.frame_id, cls, conf, bbox, area, sev, decision, self.rolling.rate)
        self.db.log_inspection(ev, latency_ms, fused, machine)
        self.latest_frame, self.latest_status = frame, {
            "event": ev.to_dict(), "category": fused.category, "urgency": fused.urgency,
            "vision_only": fused.vision_only, "latency_ms": round(latency_ms, 1), "truth": truth,
            "machine": machine.to_dict() if machine else None}

        if self.gate.should_call(fused.category, now):
            ctx = dict(category=fused.category, defect_class=cls, severity=sev, rate=self.rolling.rate,
                       window_n=self.rolling.n, zone=machine.zone if machine else None,
                       loop_ma=machine.loop_ma if machine else None)
            try:
                self.q.put_nowait(ctx)
            except queue.Full:
                pass
        return ev, fused

    # --- advisory worker -------------------------------------------------------------------
    def advise_once(self, timeout=0.0):
        try:
            ctx = self.q.get(timeout=timeout) if timeout else self.q.get_nowait()
        except queue.Empty:
            return None
        advice, meta = self.advisor.advise(ctx)
        self.latest_advice = {"category": ctx["category"], **advice, "meta": meta}
        self.db.log_advisory(time.time(), ctx["category"], advice, meta)
        return advice

    def start_worker(self):
        def loop():
            while not self._stop.is_set():
                self.advise_once(timeout=0.5)
        threading.Thread(target=loop, daemon=True).start()

    def run(self, fps=5.0):
        self.start_worker()
        period = 1.0 / fps
        while not self._stop.is_set():
            t = time.time()
            self.step()
            time.sleep(max(0.0, period - (time.time() - t)))

    def stop(self):
        self._stop.set()
