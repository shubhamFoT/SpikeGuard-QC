"""Central configuration. Every threshold lives here so you can tune it in one place
and quote the exact values in your report."""
from dataclasses import dataclass, field

CLASS_NAMES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]


@dataclass
class VisionCfg:
    imgsz: int = 256                 # NEU images are 200x200; 256 is the nearest multiple of 32
    conf_min: float = 0.25           # detections below this are ignored entirely
    reject_conf: float = 0.50        # >= this and a defect present -> reject
    # severity = bounding-box area as a fraction of the frame (assumption: tune with your mentor)
    minor_area: float = 0.02         # below 2 % of the frame  -> minor
    major_area: float = 0.10         # 2 %..10 % -> major, above 10 % -> critical


@dataclass
class FusionCfg:
    window_frames: int = 30          # rolling window for the defect rate
    min_frames: int = 10             # do not judge a rate on fewer frames than this
    rate_threshold: float = 0.30     # 30 % of recent frames defective -> "burst"
    stale_after_s: float = 3.0       # machine-health heartbeat older than this -> unavailable
    llm_cooldown_s: float = 20.0     # minimum gap between Llama calls


@dataclass
class TwinCfg:
    fs: int = 12000                  # sampling rate of the vibration data (CWRU: 12 kHz or 48 kHz)
    window_s: float = 0.5            # one analysis window
    band_lo: float = 1500.0          # AFE band-pass (models the node's analog front end)
    band_hi: float = 5000.0
    delta_frac: float = 2.0          # delta-modulator step, as a multiple of baseline noise sigma
    r: float = 0.995                 # resonator radius (decay per step); closer to 1 = narrower
    spike_k: float = 1.5             # threshold = k x largest healthy resonator peak (set at calibration)
    full_scale_hz: float = 60.0      # spike rate that maps to 20 mA
    warn_ma: float = 8.0             # >= warn_ma -> warning zone
    crit_ma: float = 16.0            # >= crit_ma -> critical zone
    fault_ma: float = 3.6            # <= this -> NAMUR NE43 style sensor/wire fault


@dataclass
class Cfg:
    vision: VisionCfg = field(default_factory=VisionCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    twin: TwinCfg = field(default_factory=TwinCfg)


CFG = Cfg()
