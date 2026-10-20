#!/usr/bin/env python3
"""Demodulación FM propia de una grabación IQ.

Entradas: .cf32 (complex64 sin cabecera, requiere --fs) o WAV de 2 canales (I, Q).

Uso:
  python fm_real.py fm_sintetico.cf32 --fs 1200000 --f0 0        # 1ª pasada: mira figures/spectrum.png
  python fm_real.py fm_sintetico.cf32 --fs 1200000 --f0 100000   # 2ª pasada: emisora centrada
Opcional: --gr compara con wfm_rcv de GNU Radio.
"""
import argparse, os
from fractions import Fraction
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_iq(path, fs_arg):
    if path.endswith('.cf32'):
        if not fs_arg:
            raise SystemExit('Con .cf32 hay que indicar --fs')
        fs, iq = fs_arg, np.fromfile(path, dtype=np.complex64)
    else:
        fs, x = wavfile.read(path)
        if x.ndim != 2 or x.shape[1] < 2:
            raise SystemExit('Se esperaba un WAV de 2 canales (I, Q)')
        iq = x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32)
        if x.dtype == np.uint8:
            iq -= (127.5 + 127.5j)
    print(f'fs={fs} Hz, muestras={len(iq)}')
    iq = iq / np.max(np.abs(iq))
    return fs, iq.astype(np.complex64)


def demod(iq, fs, f0, qr=240_000, tau=50e-6, ar=48_000):
    iq = iq * np.exp(-2j*np.pi*f0*np.arange(len(iq))/fs).astype(np.complex64)
    dec = max(1, int(fs // qr))
    iq = signal.decimate(iq, dec, ftype='fir'); fs2 = fs / dec
    d = np.angle(iq[1:] * np.conj(iq[:-1]))          # discriminador FM
    b, a = signal.bilinear([1], [tau, 1], fs2)       # desénfasis
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
    ap.add_argument('file')
    ap.add_argument('--fs', type=float, default=0, help='frecuencia de muestreo (obligatoria con .cf32)')
    ap.add_argument('--f0', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, default=0, help='limita la duración (0 = todo)')
    ap.add_argument('--gr', action='store_true')
    args = ap.parse_args()
    os.makedirs('figures', exist_ok=True); os.makedirs('data', exist_ok=True)

    fs, iq = load_iq(args.file, args.fs)
    if args.seconds:
        iq = iq[:int(fs * args.seconds)]

    plt.figure(figsize=(8, 4))
    plt.psd(iq[:int(fs*2)], NFFT=8192, Fs=fs)
    plt.title('Espectro de la grabación (centro = frecuencia central del SDR)')
    plt.savefig('figures/spectrum.png', dpi=150); plt.close()

    audio, iq_bb, fs2 = demod(iq, fs, args.f0)
    audio = audio / np.max(np.abs(audio))
    wavfile.write('data/fm_demod.wav', 48_000, (audio * 0.9).astype(np.float32))

    plt.figure(figsize=(8, 4))
    plt.specgram(audio, NFFT=2048, Fs=48_000, noverlap=1024, cmap='viridis')
    plt.xlabel('Tiempo (s)'); plt.ylabel('Frecuencia (Hz)')
    plt.title('Espectrograma del audio demodulado')
    plt.savefig('figures/audio_spectrogram.png', dpi=150); plt.close()

    # En una grabación real aquí se verían el piloto de 19 kHz y el RDS (57 kHz);
    # en la sintética solo hay la emisora y el ruido.
    plt.figure(figsize=(8, 4))
    plt.psd(iq_bb[:int(fs2*2)], NFFT=4096, Fs=fs2)
    plt.title('Emisora centrada y decimada')
    plt.savefig('figures/station_spectrum.png', dpi=150); plt.close()

    if args.gr:
        ref = gr_demod(iq_bb, fs2)
        n = min(len(ref), len(audio))
        r = np.corrcoef(ref[:n]/np.max(np.abs(ref)), audio[:n])[0, 1]
        print(f'Correlación, discriminador propio vs wfm_rcv: {r:.3f} (puede requerir alinear el retardo)')
        wavfile.write('data/fm_demod_gr.wav', 48_000, (ref/np.max(np.abs(ref))*0.9).astype(np.float32))