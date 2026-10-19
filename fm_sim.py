#!/usr/bin/env python3
"""Simulated WBFM chain with an SNR sweep (GNU Radio + SciPy).

Outputs:
  data/fm_snr_sweep.csv      snr_in_db,snr_audio_db
  figures/fm_snr_sweep.png   audio SNR versus input SNR
  figures/fm_psd_clean.png   PSD of the clean FM signal and its bandwidth (Carson)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnuradio import gr, blocks, analog
from scipy.signal import correlate, welch

AR, QR = 48_000, 240_000
SNRS_DB = [40, 30, 20, 15, 12, 10, 8, 6, 4, 2, 0]
t = np.arange(AR * 5) / AR
audio = 0.3*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*1000*t) + 0.2*np.sin(2*np.pi*3000*t)


class FmSim(gr.top_block):
    def __init__(self, noise_amp):
        super().__init__()
        src = blocks.vector_source_f(audio.astype(np.float32).tolist(), False)
        tx = analog.wfm_tx(AR, QR, tau=50e-6, max_dev=75e3)
        noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amp, 1)
        head = blocks.head(gr.sizeof_gr_complex, len(audio) * QR // AR)
        add = blocks.add_cc()
        rx = analog.wfm_rcv(QR, QR // AR)
        self.clean, self.nz, self.out = blocks.vector_sink_c(), blocks.vector_sink_c(), blocks.vector_sink_f()
        self.connect(src, tx)
        self.connect(tx, (add, 0)); self.connect(tx, self.clean)
        self.connect(noise, head, (add, 1)); self.connect(head, self.nz)
        self.connect(add, rx, self.out)


def audio_snr(ref, rec):
    n = min(len(ref), len(rec)); ref, rec = ref[:n], rec[:n]
    d = np.argmax(correlate(rec, ref, mode='full', method='fft')) - (n - 1)
    if d > 0: rec = rec[d:]; ref = ref[:len(rec)]
    elif d < 0: ref = ref[-d:]; rec = rec[:len(ref)]
    k = AR // 2; ref, rec = ref[k:-k], rec[k:-k]
    g = np.dot(rec, ref) / np.dot(rec, rec)
    return 10*np.log10(np.sum(ref**2) / np.sum((ref - g*rec)**2))


def sweep():
    rows, clean = [], None
    for snr in SNRS_DB:
        tb = FmSim(10 ** (-snr / 20)); tb.run()
        clean, nz = np.array(tb.clean.data()), np.array(tb.nz.data())
        snr_in = 10*np.log10(np.mean(abs(clean)**2) / np.mean(abs(nz)**2))
        snr_out = audio_snr(audio, np.array(tb.out.data()))
        print(f"{snr_in:.1f},{snr_out:.1f}")
        rows.append((snr_in, snr_out))
    return np.array(rows), clean


def plot_sweep(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rows[:, 0], rows[:, 1], 'o-')
    ax.set_xlabel('Input SNR (dB)'); ax.set_ylabel('Audio SNR (dB)')
    ax.set_title('Wideband FM: audio SNR versus input SNR')
    ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig('figures/fm_snr_sweep.png', dpi=150)


def plot_psd(clean):
    f, p = welch(clean, fs=QR, nperseg=4096, return_onesided=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    pdb = 10*np.log10(p / p.max())
    # bandwidth at -20 dB as an estimate of the occupied bandwidth
    inside = f[pdb > -20]
    bw = inside.max() - inside.min()
    print(f"Bandwidth at -20 dB: {bw/1e3:.0f} kHz (Carson: 180 kHz)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(f/1e3, pdb)
    ax.axvline(-90, ls='--', c='gray'); ax.axvline(90, ls='--', c='gray', label='Carson ±90 kHz')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('PSD (dB rel.)'); ax.set_ylim(-80, 3)
    ax.set_title(f'PSD of the clean FM signal (-20 dB bandwidth ≈ {bw/1e3:.0f} kHz)')
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig('figures/fm_psd_clean.png', dpi=150)


if __name__ == '__main__':
    rows, clean = sweep()
    np.savetxt('data/fm_snr_sweep.csv', rows, delimiter=',', fmt='%.2f',
               header='snr_in_db,snr_audio_db', comments='')
    plot_sweep(rows)
    plot_psd(clean)
