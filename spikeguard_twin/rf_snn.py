"""Behavioural model (a 'digital twin') of the SpikeGuard signal chain:

   vibration -> AFE band-pass -> delta modulator -> Resonate-and-Fire bank -> spike rate -> mA

It is NOT the RTL. It is a software reference that you later verify the Verilog against.
Parameters here are my assumptions, not values taken from your hardware design."""
import numpy as np
from scipy.signal import butter, sosfilt
from config import CFG
from common.schemas import MachineState
from spikeguard_twin.signals import ORDERS

C = CFG.twin


def afe_bandpass(x, fs, lo=None, hi=None):
    sos = butter(2, [lo or C.band_lo, hi or C.band_hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, x)


def delta_modulate(x, delta):
    """Sampled approximation of a continuous-time delta modulator: emit +1/-1 whenever the
    input moves more than `delta` away from a staircase reference, and move the reference."""
    ev = np.zeros(len(x), dtype=np.int8)
    ref = x[0]
    for i in range(1, len(x)):
        d = x[i] - ref
        if d > delta:
            ev[i] = 1
            ref += delta
        elif d < -delta:
            ev[i] = -1
            ref -= delta
    return ev


def estimate_shaft_freq(x, fs, lo=10.0, hi=100.0):
    """Software stand-in for the tacholess PLL: pick the strongest spectral line in the
    shaft-speed range of the UNFILTERED signal."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    band = (freqs >= lo) & (freqs <= hi)
    return float(freqs[band][np.argmax(spec[band])])


def csd_terms(value, frac_bits=12):
    """Number of signed power-of-two terms (canonical signed digit) needed for a coefficient.
    Fewer terms = fewer adders in hardware = a multiplier-free implementation."""
    n = int(round(abs(value) * (1 << frac_bits)))
    terms = 0
    while n:
        if n & 1:
            terms += 1
            n = n + 1 if (n & 3) == 3 else n - 1
        n >>= 1
    return terms


class RFBank:
    """A bank of Resonate-and-Fire neurons: z[n+1] = r*exp(j*theta)*z[n] + u[n].
    Neuron k is tuned to f_k = order_k * shaft_freq (that is the 'speed adaptation')."""

    def __init__(self, fs, shaft_hz, orders=None, r=None):
        self.fs = fs
        self.names = list((orders or ORDERS).keys())
        self.freqs = np.array([(orders or ORDERS)[k] * shaft_hz for k in self.names])
        theta = 2 * np.pi * self.freqs / fs
        self.r = r or C.r
        self.a = self.r * np.cos(theta)     # real part of the pole
        self.b = self.r * np.sin(theta)     # imaginary part of the pole
        self.thr = None                     # set by calibrate()

    def run(self, u, thr=None):
        """Drive the bank with input u (1-D). Returns (spike_count_per_neuron, peak |Im z|)."""
        n_neu = len(self.freqs)
        re = np.zeros(n_neu)
        im = np.zeros(n_neu)
        spikes = np.zeros(n_neu, dtype=int)
        peak = np.zeros(n_neu)
        thr_arr = np.full(n_neu, np.inf) if thr is None else np.asarray(thr)
        for s in u:
            re, im = self.a * re - self.b * im + s, self.b * re + self.a * im
            peak = np.maximum(peak, np.abs(im))
            fired = im >= thr_arr
            spikes += fired
            re[fired] = 0.0                 # reset on spike
            im[fired] = 0.0
        return spikes, peak


class SpikeGuardTwin:
    """Calibrate on healthy windows, then convert each new window into a MachineState."""

    def __init__(self, fs=None):
        self.fs = fs or C.fs
        self.bank = None
        self.sigma = None
        self.delta = None
        self.mean_act = 0.0
        self.thr = None
        self.base_pk = None
        self.shaft_hz = None

    def _front_end(self, x):
        xb = afe_bandpass(x, self.fs)
        if self.sigma is None:                      # fixed at calibration, never changed after
            self.sigma = float(np.std(xb))
            self.delta = C.delta_frac * self.sigma
        return np.abs(delta_modulate(xb, self.delta)).astype(float)

    def calibrate(self, healthy_windows, shaft_hz):
        """Learn the healthy baseline at commissioning: the delta step, the mean event rate and
        each neuron's firing threshold all come from healthy data, so no manual tuning is needed."""
        self.shaft_hz = shaft_hz
        self.bank = RFBank(self.fs, shaft_hz)
        self.sigma = None
        acts = [self._front_end(w) for w in healthy_windows]
        self.mean_act = float(np.mean([a.mean() for a in acts]))
        peaks = np.array([self.bank.run(a - self.mean_act)[1] for a in acts])
        self.base_pk = peaks.max(axis=0) + 1e-9
        self.thr = C.spike_k * self.base_pk          # fire only well above anything seen when healthy
        return self

    def process(self, x, shaft_hz=None, ts=0.0):
        if shaft_hz is not None and abs(shaft_hz - self.shaft_hz) > 0.5:
            self.shaft_hz = shaft_hz                 # speed adaptation: re-tune neurons to new order
            self.bank = RFBank(self.fs, shaft_hz)
        act = self._front_end(x)
        spikes, pk = self.bank.run(act - self.mean_act, self.thr)
        rate = float(spikes.sum() / (len(x) / self.fs))
        ma = 4.0 + 16.0 * min(rate / C.full_scale_hz, 1.0)
        zone = "critical" if ma >= C.crit_ma else "warning" if ma >= C.warn_ma else "healthy"
        hi = float(np.max(pk / self.base_pk))
        return MachineState(ts=ts, loop_ma=round(ma, 2), zone=zone, spike_rate_hz=round(rate, 1),
                            health_index=round(hi, 2), source="simulated")
