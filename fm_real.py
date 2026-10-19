#!/usr/bin/env python3
"""Custom FM demodulation of an IQ recording (2-channel WAV: I, Q).

Usage:
  python fm_real.py data/fm_rds.wav --f0 0          # 1st pass: check figures/spectrum.png
  python fm_real.py data/fm_rds.wav --f0 -250000    # 2nd pass: with the station centered
Optional: --gr  compares against GNU Radio's wfm_rcv.
"""
import argparse
from fractions import Fraction
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_iq(path):
    fs, x = wavfile.read(path)
    print(f"fs={fs} Hz, shape={x.shape}, dtype={x.dtype}")
    if x.ndim != 2 or x.shape[1] < 2:
        raise SystemExit("Expected a 2-channel (I, Q) WAV")
    iq = x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32)
    if x.dtype == np.uint8:
        iq -= (127.5 + 127.5j)
    iq /= np.max(np.abs(iq))
    return fs, iq.astype(np.complex64)


def demod(iq, fs, f0, qr=240_000, tau=50e-6, ar=48_000):
    iq = iq * np.exp(-2j*np.pi*f0*np.arange(len(iq))/fs).astype(np.complex64)
    dec = max(1, int(fs // qr))
    iq = signal.decimate(iq, dec, ftype='fir'); fs2 = fs / dec
    d = np.angle(iq[1:] * np.conj(iq[:-1]))          # FM discriminator
    b, a = signal.bilinear([1], [tau, 1], fs2)       # de-emphasis
    audio = signal.lfilter(b, a, d)
    fr = Fraction(ar / fs2).limit_denominator(1000)
    audio = signal.resample_poly(audio, fr.numerator, fr.denominator)
    return audio, iq, fs2


def gr_demod(iq, fs2, ar=48_000):
    from gnuradio import gr, blocks, analog
    tb = gr.top_block()
    src = blocks.vector_source_c(iq.tolist(), False)
    rx = analog.wfm_rcv(fs2, int(fs2 // ar))
    snk = blocks.vector_sink_f()
    tb.connect(src, rx, snk); tb.run()
    return np.array(snk.data())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('wav'); ap.add_argument('--f0', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, default=0, help='limit the duration (0 = all)')
    ap.add_argument('--gr', action='store_true')
    args = ap.parse_args()

    fs, iq = load_iq(args.wav)
    if args.seconds:
        iq = iq[:int(fs * args.seconds)]

    plt.figure(figsize=(8, 4))
    plt.psd(iq[:int(fs*2)], NFFT=8192, Fs=fs)
    plt.title('Recording spectrum (center = SDR center frequency)')
    plt.savefig('figures/spectrum.png', dpi=150); plt.close()

    audio, iq_bb, fs2 = demod(iq, fs, args.f0)
    audio = audio / np.max(np.abs(audio))
    wavfile.write('data/fm_demod.wav', 48_000, (audio * 0.9).astype(np.float32))

    plt.figure(figsize=(8, 4))
    plt.specgram(audio, NFFT=2048, Fs=48_000, noverlap=1024, cmap='viridis')
    plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)'); plt.title('Spectrogram of the demodulated audio')
    plt.savefig('figures/audio_spectrogram.png', dpi=150); plt.close()

    # PSD after centering and decimating: shows the station, the 19 kHz pilot and the RDS (57 kHz)
    plt.figure(figsize=(8, 4))
    plt.psd(iq_bb[:int(fs2*2)], NFFT=4096, Fs=fs2)
    plt.title('Station centered and decimated'); plt.savefig('figures/station_spectrum.png', dpi=150); plt.close()

    if args.gr:
        ref = gr_demod(iq_bb, fs2)
        n = min(len(ref), len(audio))
        r = np.corrcoef(ref[:n]/np.max(np.abs(ref)), audio[:n])[0, 1]
        print(f"Correlation, custom discriminator vs wfm_rcv: {r:.3f} (may need delay alignment)")
        wavfile.write('data/fm_demod_gr.wav', 48_000, (ref/np.max(np.abs(ref))*0.9).astype(np.float32))
