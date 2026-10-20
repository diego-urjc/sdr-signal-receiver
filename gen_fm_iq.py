#!/usr/bin/env python3
"""Generates a synthetic wideband IQ recording with three FM stations plus noise.

Output: data/fm_sintetico.cf32 (complex64, 1.2 MS/s, 8 s, ~77 MB; not tracked by git).
"""
import os
import numpy as np
from scipy import signal

FS_IQ, FS_AU, DUR = 1_200_000, 48_000, 8
DEV, TAU = 75e3, 50e-6


def make_audios():
    """Station offset from the centre frequency (Hz) -> original audio."""
    t = np.arange(FS_AU * DUR) / FS_AU
    return {
        -400e3: signal.chirp(t, f0=300, t1=DUR, f1=3000),                    # sweep
         100e3: 0.5*np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*660*t),       # two tones
         350e3: np.sin(2*np.pi*1000*t) * (np.sin(2*np.pi*2*t) > 0),          # beeps
    }


def fm_mod(audio):
    b, a = signal.bilinear([TAU, 1], [TAU/10, 1], FS_AU)   # approximate pre-emphasis
    x = signal.lfilter(b, a, audio)
    x = 0.9 * x / np.max(np.abs(x))
    x = signal.resample_poly(x, FS_IQ // FS_AU, 1)         # 48 kHz -> 1.2 MHz
    return np.exp(2j*np.pi*DEV*np.cumsum(x)/FS_IQ)


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    n = FS_IQ * DUR
    k = np.arange(n)
    iq = np.zeros(n, np.complex64)
    for f_off, audio in make_audios().items():
        iq += (fm_mod(audio) * np.exp(2j*np.pi*f_off*k/FS_IQ)).astype(np.complex64)
    iq += (0.05*(rng.standard_normal(n) + 1j*rng.standard_normal(n))).astype(np.complex64)
    os.makedirs('data', exist_ok=True)
    iq.tofile('data/fm_sintetico.cf32')
