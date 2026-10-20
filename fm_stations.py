#!/usr/bin/env python3
"""Demodulates the three stations of the synthetic recording and measures audio quality.

Usage: python fm_stations.py [data/fm_sintetico.cf32]
Output: figures/spectrogram_<station>.png and data/fm_station_snr.csv
Audio SNR is measured against the known original audio (delay and gain aligned);
the pre-emphasis used by gen_fm_iq.py is only approximate, so values are a lower bound.
"""
import csv
import sys
import numpy as np
from scipy.signal import correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fm_real import load_iq, demod
from gen_fm_iq import make_audios, FS_IQ

AR = 48_000
NAMES = {-400e3: 'm400k', 100e3: 'p100k', 350e3: 'p350k'}


def audio_snr(ref, rec):
    n = min(len(ref), len(rec)); ref, rec = ref[:n], rec[:n]
    d = np.argmax(correlate(rec, ref, mode='full', method='fft')) - (n - 1)
    if d > 0: rec = rec[d:]; ref = ref[:len(rec)]
    elif d < 0: ref = ref[-d:]; rec = rec[:len(ref)]
    k = AR // 2; ref, rec = ref[k:-k], rec[k:-k]
    g = np.dot(rec, ref) / np.dot(rec, rec)
    return 10*np.log10(np.sum(ref**2) / np.sum((ref - g*rec)**2))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/fm_sintetico.cf32'
    fs, iq = load_iq(path, FS_IQ)
    rows = []
    for f0, ref in make_audios().items():
        audio, _, _ = demod(iq, fs, f0)
        audio = audio / np.max(np.abs(audio))
        snr = audio_snr(ref / np.max(np.abs(ref)), audio)
        rows.append((f'{f0/1e3:+.0f}', f'{snr:.1f}'))
        print(f'{f0/1e3:+.0f} kHz: audio SNR {snr:.1f} dB')
        plt.figure(figsize=(6, 4))
        plt.specgram(audio, NFFT=2048, Fs=AR, noverlap=1024, cmap='viridis')
        plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
        plt.title(f'Demodulated audio, station {f0/1e3:+.0f} kHz')
        plt.tight_layout(); plt.savefig(f'figures/spectrogram_{NAMES[f0]}.png', dpi=130); plt.close()
    with open('data/fm_station_snr.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['station_offset_khz', 'audio_snr_db']); w.writerows(rows)
