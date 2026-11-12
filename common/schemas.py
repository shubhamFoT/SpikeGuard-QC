"""The interface contract between the two halves of the system (Step 1 of the guide)."""
from dataclasses import dataclass, asdict
from typing import List, Optional

ZONES = ("healthy", "warning", "critical", "fault")
ZONE_CODE = {name: i for i, name in enumerate(ZONES)}
SEVERITIES = ("none", "minor", "major", "critical")


@dataclass
class MachineState:
    ts: float
    loop_ma: float
    zone: str                 # one of ZONES
    spike_rate_hz: float
    health_index: float
    source: str = "simulated"  # "simulated" now, "hardware" for the real SpikeGuard node

    def to_dict(self):
        return asdict(self)


@dataclass
class InspectionEvent:
    ts: float
    frame_id: int
    defect_class: Optional[str]
    confidence: float
    bbox: Optional[List[float]]      # [x1, y1, x2, y2] in pixels
    area_frac: float
    severity: str                    # one of SEVERITIES
    decision: str                    # "pass" | "reject" | "review"
    rolling_defect_rate: float

    def to_dict(self):
        return asdict(self)
