#!/usr/bin/env python3
"""Builds two extra README figures: figures/adsb_frame.png and figures/banner.png.

Usage: python make_figures.py   (needs data/modes1.bin and the figures from the other scripts)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from adsb_demod import demodulate, crc24

# --- one decoded ADS-B frame: magnitude with preamble and data bits marked ---
raw = np.fromfile('data/modes1.bin', np.uint8)
mag = np.abs((raw[0::2] - 127.5) + 1j * (raw[1::2] - 127.5))
frames = demodulate('data/modes1.bin')
target = bytes.fromhex(frames[0])
bits = np.unpackbits(np.frombuffer(target, np.uint8))
n = len(mag) - 240
for p in range(n):
    d = mag[p+16:p+240]
    if np.array_equal((d[0::2] > d[1::2]).astype(np.uint8), bits) and crc24(bits) == 0:
        break
seg = mag[p:p+240]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(seg, drawstyle='steps-post', lw=1)
for s in (0, 2, 7, 9):
    ax.axvspan(s, s+1, color='tab:red', alpha=.3)
ax.axvspan(16, 240, color='tab:green', alpha=.07)
ax.set_xlabel('Sample (2 MS/s)'); ax.set_ylabel('Magnitude')
ax.set_title(f'ADS-B frame {frames[0].upper()}: preamble pulses (red) and 112 data bits (green)', fontsize=9)
fig.tight_layout(); fig.savefig('figures/adsb_frame.png', dpi=140); plt.close(fig)

# --- banner: three key figures side by side ---
imgs = [('SNR threshold effect', 'figures/fm_threshold.png'),
        ('Three FM stations in the IQ recording', 'figures/spectrum.png'),
        ('Custom ADS-B demodulator', 'figures/adsb_frame.png')]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, f) in zip(axs, imgs):
    ax.imshow(mpimg.imread(f)); ax.set_title(title); ax.axis('off')
fig.tight_layout(); fig.savefig('figures/banner.png', dpi=130); plt.close(fig)

# --- threshold plot in English, rebuilt from the sweep CSV (same data as fm_sim.py) ---
import pandas as pd
df = pd.read_csv('data/fm_snr_sweep.csv')
plt.figure(figsize=(8, 5))
plt.plot(df.SNR_Input, df.SNR_Audio, 'o-', color='blue', label='FM demodulation curve')
plt.grid(True, linestyle='--'); plt.gca().invert_xaxis(); plt.legend()
plt.xlabel('Input SNR (RF / channel) [dB]'); plt.ylabel('Audio SNR (demodulated output) [dB]')
plt.title('FM threshold effect (SNR curve)')
plt.savefig('figures/fm_threshold.png', dpi=130)
