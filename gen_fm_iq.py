#!/usr/bin/env python3
import numpy as np
from scipy import signal

FS_IQ, FS_AU, DUR = 1_200_000, 48_000, 8
DEV, TAU = 75e3, 50e-6
rng = np.random.default_rng(1)
t = np.arange(FS_AU * DUR) / FS_AU

# offset de cada emisora respecto al centro -> audio
audios = {
    -400e3: signal.chirp(t, f0=300, t1=DUR, f1=3000),                    # barrido
     100e3: 0.5*np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*660*t),       # dos tonos
     350e3: np.sin(2*np.pi*1000*t) * (np.sin(2*np.pi*2*t) > 0),          # pitidos
}

def fm_mod(audio):
    b, a = signal.bilinear([TAU, 1], [TAU/10, 1], FS_AU)   # pre-énfasis aproximado
    x = signal.lfilter(b, a, audio)
    x = 0.9 * x / np.max(np.abs(x))
    x = signal.resample_poly(x, FS_IQ // FS_AU, 1)         # 48 kHz -> 1,2 MHz
    return np.exp(2j*np.pi*DEV*np.cumsum(x)/FS_IQ)

n = FS_IQ * DUR
k = np.arange(n)
iq = np.zeros(n, np.complex64)
for f_off, audio in audios.items():
    iq += (fm_mod(audio) * np.exp(2j*np.pi*f_off*k/FS_IQ)).astype(np.complex64)
iq += (0.05*(rng.standard_normal(n) + 1j*rng.standard_normal(n))).astype(np.complex64)
iq.tofile('fm_sintetico.cf32')
