"""Vibration sources for the SpikeGuard digital twin: a synthetic generator (works with no
dataset, used in tests and as a demo fallback) and a CWRU .mat loader."""
import numpy as np

# CWRU 6205-2RS JEM SKF drive-end bearing: fault frequency / shaft frequency (published multipliers)
ORDERS = {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 4.7135}


def synth_vibration(fs=12000, seconds=0.5, rpm=1797.0, fault=None, severity=1.0,
                    f_res=3000.0, seed=0):
    """Healthy: shaft harmonics + broadband noise.
    Faulty: add an impulse train at the fault frequency; each impulse rings a structural
    resonance at f_res (this is the textbook bearing-fault signature)."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    fr = rpm / 60.0
    x = 0.20 * np.sin(2 * np.pi * fr * t) + 0.08 * np.sin(2 * np.pi * 2 * fr * t)
    x += 0.05 * rng.standard_normal(n)
    if fault:
        f_fault = ORDERS[fault] * fr
        ring_t = np.arange(int(0.004 * fs)) / fs
        ring = np.exp(-900.0 * ring_t) * np.sin(2 * np.pi * f_res * ring_t)
        k = 0.0
        while True:
            k += (1.0 / f_fault) * (1.0 + 0.01 * rng.standard_normal())   # ~1 % slip jitter
            i = int(k * fs)
            if i >= n:
                break
            seg = ring[: n - i]
            x[i:i + len(seg)] += severity * 0.35 * (0.8 + 0.4 * rng.random()) * seg
    return x.astype(np.float64), rpm


def load_cwru(path, channel="DE"):
    """Load a Case Western Reserve University .mat file. Returns (signal, rpm_or_None).
    Download the files yourself from the CWRU Bearing Data Center."""
    from scipy.io import loadmat
    m = loadmat(path)
    sig_key = next(k for k in m if k.endswith(f"_{channel}_time"))
    rpm_key = next((k for k in m if k.endswith("RPM")), None)
    rpm = float(np.ravel(m[rpm_key])[0]) if rpm_key else None
    return np.ravel(m[sig_key]).astype(np.float64), rpm


def windows(x, fs, window_s):
    """Yield consecutive non-overlapping windows."""
    n = int(fs * window_s)
    for i in range(0, len(x) - n + 1, n):
        yield x[i:i + n]
