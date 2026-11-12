"""Deterministic fusion of vision results with machine health. No AI in this file on purpose:
every advisory category must be explainable, testable and auditable."""
from collections import deque
from dataclasses import dataclass
from typing import Optional

from config import CFG
from common.schemas import MachineState

F = CFG.fusion

NORMAL = "NORMAL"
ISOLATED_DEFECT = "ISOLATED_DEFECT"
PROCESS_OR_MATERIAL = "PROCESS_OR_MATERIAL_SUSPECT"
MACHINE_WEAR = "MACHINE_WEAR_SUSPECT"
MACHINE_WARNING_ONLY = "MACHINE_WARNING_NO_DEFECT_TREND"
STOP_AND_INSPECT = "STOP_AND_INSPECT"
HEALTH_UNAVAILABLE = "MACHINE_HEALTH_UNAVAILABLE"

CATEGORY_URGENCY = {NORMAL: "low", ISOLATED_DEFECT: "low", PROCESS_OR_MATERIAL: "medium",
                    MACHINE_WEAR: "medium", MACHINE_WARNING_ONLY: "low",
                    STOP_AND_INSPECT: "high", HEALTH_UNAVAILABLE: "medium"}


class RollingRate:
    """Fraction of the last N frames that contained a defect."""

    def __init__(self, n=None):
        self.buf = deque(maxlen=n or F.window_frames)

    def add(self, defective: bool):
        self.buf.append(1 if defective else 0)

    @property
    def n(self):
        return len(self.buf)

    @property
    def rate(self):
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

    @property
    def burst(self):
        return self.n >= F.min_frames and self.rate >= F.rate_threshold


@dataclass
class Fused:
    category: str
    urgency: str
    vision_only: bool
    note: str


def fuse(rolling: RollingRate, last_frame_defective: bool,
         machine: Optional[MachineState]) -> Fused:
    """Implements the table in section 4.3 of the guide. `machine=None` means the reader reported
    stale/unreachable data, which is treated exactly like a sensor fault (graceful degradation)."""
    unavailable = machine is None or machine.zone == "fault"
    burst = rolling.burst

    if unavailable:
        note = "defect burst, machine health unknown" if burst else "machine health unknown"
        urgency = "medium" if burst else "low"
        return Fused(HEALTH_UNAVAILABLE, urgency, True, note)

    if machine.zone == "critical":
        return Fused(STOP_AND_INSPECT, "high", False, "machine health critical")

    if burst and machine.zone == "warning":
        return Fused(MACHINE_WEAR, "medium", False, "defect burst while bearing health is in warning")
    if burst:
        return Fused(PROCESS_OR_MATERIAL, "medium", False, "defect burst while machine is healthy")
    if machine.zone == "warning":
        return Fused(MACHINE_WARNING_ONLY, "low", False, "bearing warning, no defect trend yet")
    if last_frame_defective:
        return Fused(ISOLATED_DEFECT, "low", False, "isolated defect, machine healthy")
    return Fused(NORMAL, "low", False, "normal operation")


class AdvisoryGate:
    """Decide WHEN to call Llama: only on a change into a non-NORMAL category, and not more often
    than the cooldown. This keeps the slow model off the real-time path."""

    def __init__(self, cooldown_s=None):
        self.cooldown = cooldown_s if cooldown_s is not None else F.llm_cooldown_s
        self.prev = NORMAL
        self.last_call = -1e18

    def should_call(self, category: str, now: float) -> bool:
        changed = category != self.prev
        self.prev = category
        if category in (NORMAL,) or not changed:
            return False
        if now - self.last_call < self.cooldown:
            return False
        self.last_call = now
        return True
