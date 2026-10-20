Práctica: recepción y análisis de señales SDR por software

Esta práctica trabaja sobre señales simuladas y grabaciones IQ, sin hardware. Reutiliza el Ubuntu en máquina virtual del laboratorio anterior.

Objetivos
Simular una cadena FM de banda ancha y medir cómo cae la calidad del audio al bajar la SNR.
Demodular una grabación FM real y analizar su espectro.
Decodificar mensajes ADS-B desde un fichero IQ y comparar con tu propio demodulador.
Documentar todo en un informe técnico y un repositorio.
Fundamento
IQ: muestras complejas (I + jQ) del espectro alrededor de una frecuencia central.
FM: la información va en la frecuencia instantánea. La FM de radiodifusión usa 75 kHz de desviación máxima y pre-énfasis de 50 µs en Europa. La regla de Carson da un ancho de banda de ≈ 2·(75 + 15) = 180 kHz.
ADS-B: modulación PPM a 1 Mbit/s en 1090 MHz, con tramas de 112 bits.
Fechas
Fecha límite	Entregable
Dom 25 oct	Entorno instalado y FM simulada funcionando
Sáb 31 oct	Barrido de SNR con gráfica
Dom 1 nov	Demodulación de la grabación FM real
Mié 4 nov	ADS-B decodificado con dump1090 y pyModeS
Vie 6 nov	Demodulador propio (reto, opcional)
Sáb 7 nov	Informe con gráficas
Dom 8 nov	Repo publicado
Parte 0. Entorno
bash
sudo apt install -y gnuradio python3-numpy python3-scipy python3-matplotlib python3-pandas \
  build-essential git librtlsdr-dev libusb-1.0-0-dev pkg-config
python3 -m venv --system-site-packages venv && source venv/bin/activate
pip install pyModeS

El --system-site-packages hace que el entorno virtual vea el GNU Radio instalado con apt. Comprueba con python3 -c "from gnuradio import analog; print('ok')".

Parte 1. FM simulada (hasta el domingo 25 y el sábado 31)

Paso 1, flujo en GNU Radio Companion (para capturas del informe). Crea esta cadena: fuente de señal → WBFM Transmit → Channel Model → WBFM Receive → Audio Sink, y añade un QT GUI Frequency Sink tras el Channel Model para ver el espectro. WBFM Receive hace demodulación mono con desénfasis y admite desviación y tau como parámetros. El Channel Model añade ruido AWGN, desplazamiento de frecuencia y multipath. Guarda el .grc y las capturas.

Paso 2, versión en Python para el barrido (fm_sim.py). Usamos una fuente de ruido propia para controlar la SNR con exactitud:

python
#!/usr/bin/env python3
import numpy as np
from gnuradio import gr, blocks, analog
from scipy.signal import correlate

AR, QR = 48_000, 240_000
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

if __name__ == '__main__':
    for snr in [40, 30, 20, 15, 12, 10, 8, 6, 4, 2, 0]:
        tb = FmSim(10 ** (-snr / 20)); tb.run()
        clean, nz = np.array(tb.clean.data()), np.array(tb.nz.data())
        snr_in = 10*np.log10(np.mean(abs(clean)**2) / np.mean(abs(nz)**2))
        print(f"{snr_in:.1f},{audio_snr(audio, np.array(tb.out.data())):.1f}")

Las firmas de los bloques pueden variar según tu versión de GNU Radio. Si algo falla, mira help(analog.wfm_tx).

Paso 3, resultado. Guarda la salida como CSV y haz la gráfica “SNR de audio frente a SNR de entrada”. Debería aparecer un codo claro a bajas SNR, el efecto umbral de la FM. Explícalo en el informe.

Paso 4, espectro. Con scipy.signal.welch, calcula la densidad espectral de la señal limpia y comprueba que su ancho está cerca de los 180 kHz de la regla de Carson.

Parte 2. FM real grabada (domingo 1 de noviembre)

Paso 1, descarga. SDRplay publica grabaciones IQ de ejemplo, entre ellas una de FM con RDS, empaquetadas en WAV. Mira sus condiciones y la nota legal de su página. No he podido comprobar el formato exacto del fichero, así que verifica el número de canales y la frecuencia de muestreo al abrirlo.

Paso 2, demodulación propia. fm_real.py:

python
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

fs, x = wavfile.read('fm_rds.wav')            # esperado: (N, 2) = I, Q
iq = x[:, 0].astype(np.float32) + 1j * x[:, 1].astype(np.float32)
iq /= np.max(np.abs(iq))

plt.psd(iq[:fs*2], NFFT=8192, Fs=fs); plt.savefig('espectro.png')   # busca la emisora
f0 = 0.0                                       # Hz respecto al centro; ajústalo viendo el espectro
iq *= np.exp(-2j*np.pi*f0*np.arange(len(iq))/fs)

dec = max(1, int(fs // 240_000))
iq = signal.decimate(iq, dec, ftype='fir'); fs2 = fs / dec
d = np.angle(iq[1:] * np.conj(iq[:-1]))        # discriminador FM
b, a = signal.bilinear([1], [50e-6, 1], fs2)   # desénfasis de 50 µs
audio = signal.lfilter(b, a, d)
audio = signal.resample_poly(audio, 48_000, int(fs2))
wavfile.write('fm_demod.wav', 48_000, (audio / np.max(np.abs(audio)) * 0.9).astype(np.float32))

Si el fichero trae un canal extra o scipy protesta por el formato, prueba con soundfile.

Paso 3, para el informe. Guarda el espectro con la emisora localizada, el audio demodulado y el espectrograma del audio. Compara tu discriminador con la cadena WBFM Receive de GNU Radio.

Parte 3. ADS-B desde fichero (miércoles 4 de noviembre)

Paso 1, dump1090. El repo de dump1090 incluye un fichero de prueba testfiles/modes1.bin y el programa decodifica IQ de 8 bits sin signo a 2 MHz con --ifile.

bash
git clone https://github.com/antirez/dump1090.git && cd dump1090 && make
./dump1090 --ifile testfiles/modes1.bin --raw > ../frames.txt

Paso 2, análisis con pyModeS, que decodifica mensajes ADS-B (DF17):

python
import pyModeS as pms
rows = []
for line in open('frames.txt'):
    msg = line.strip().strip('*;')
    if len(msg) != 28 or pms.df(msg) != 17 or pms.crc(msg) != 0:
        continue
    tc = pms.adsb.typecode(msg)
    row = {'icao': pms.adsb.icao(msg), 'tc': tc}
    if 1 <= tc <= 4:   row['callsign'] = pms.adsb.callsign(msg)
    elif 9 <= tc <= 18: row['alt_ft'] = pms.adsb.altitude(msg)
    elif tc == 19:      row['vel'] = pms.adsb.velocity(msg)
    rows.append(row)

Esa API es la de pyModeS 2.x. Si tu versión es más nueva, consulta mode-s.org/api. Saca una tabla por aeronave con indicativo, altitud y velocidad.

Parte 4. Reto: demodulador propio (viernes 6)

Opcional, pero es lo que más se valora.

Datos y pasos:

Lee el fichero: raw = np.fromfile('modes1.bin', np.uint8), iq = (raw[0::2]-127.5) + 1j*(raw[1::2]-127.5), y trabaja con la magnitud mag = np.abs(iq).
A 2 Msps, cada bit son 2 muestras. El preámbulo ocupa 16 muestras, con pulsos en las posiciones 0, 2, 7 y 9, y los datos empiezan en la 16.
Para cada bit, 1 si mag[16+2k] > mag[16+2k+1] y 0 en caso contrario. Decodifica 112 bits.
Comprueba el CRC de 24 bits con este divisor:
python
def crc24(bits):                  # bits: lista de 112 enteros 0/1
    reg = 0
    for b in bits:
        reg = (reg << 1) | b
        if reg & (1 << 24): reg ^= 0x1FFF409
    return reg                    # 0 = trama válida
Convierte las tramas válidas a hexadecimal y compara con frames.txt: cuántas encuentra cada uno y cuántas coinciden. Sé honesto en el informe sobre las diferencias.
Parte 5. Informe y repo (7 y 8 de noviembre)

Informe de 4 a 6 páginas: resumen, introducción, método, resultados con las gráficas, discusión (efecto umbral, limitaciones), conclusiones y referencias.

Repo:

README.md  fm_sim.py  fm_real.py  adsb_decode.py  adsb_demod.py
flowgraphs/ (.grc)   data/ (CSV)   figures/   report.pdf

Limitaciones que debe reconocer el README: simulación y grabaciones, no captura propia; ruido AWGN ideal; sin sincronización de reloj real; ficheros de terceros con licencias propias (no los subas, enlázalos). Si quieres compartir tus grabaciones IQ con metadatos, el formato es SigMF.

Línea para el CV

“Simulación y análisis de radio definida por software con GNU Radio y Python: cadena FM con barrido de SNR, demodulación de grabaciones IQ reales y decodificación ADS-B con demodulador propio.”

Pásame los errores o las salidas de cada parte y te ayudo a depurar.

Sources: GNU Radio, WBFM Receive · GNU Radio, Channel Model · dump1090, uso con ficheros · pyModeS · SDRplay, grabaciones IQ · IQEngine y SigMF

Entiendo el problema. La página de SDRplay que puedo ver sigue listando la grabación de FM con RDS (39 MB), pero no puedo comprobar que el enlace de descarga funcione, y es posible que esté caído. Puedes intentar dos cosas rápidas, pero no dependas de ellas.

Opciones
Reintentar SDRplay (5 minutos). Prueba la descarga en otro navegador. Si da error, descártala.
IQEngine (iqengine.org). Su instancia principal funciona como repositorio de grabaciones SigMF de ejemplo, así que busca ahí alguna de FM de radiodifusión. No he podido confirmar que haya una, así que no cuentes con ello.
Generar tú la grabación (recomendada). Siempre funciona y no depende de nadie. Creas un fichero IQ de banda ancha con 3 emisoras FM sintéticas a distintas frecuencias, con ruido, y practicas exactamente la misma cadena: localizar la emisora en el espectro, bajarla a banda base, filtrar y decimar, y demodular.

Si usas la 3, el README debe decir claramente que el fichero es sintético, no una captura real. A cambio, como conoces el audio original de cada emisora, puedes medir la calidad de tu demodulación, algo que con una grabación real no podrías.

1. Generar la grabación (gen_fm_iq.py)
python
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

Ocupa unos 77 MB. No lo subas al repo, deja solo el script.

2. Cambios en fm_real.py

Sustituye las líneas que leían el WAV por esto y deja el resto igual:

python
fs = 1_200_000
iq = np.fromfile('fm_sintetico.cf32', dtype=np.complex64)
plt.psd(iq[:fs*2], NFFT=8192, Fs=fs); plt.savefig('espectro.png')
f0 = 100e3     # prueba -400e3, 100e3 y 350e3

El resto del script no cambia: 1,2 MHz entre 5 da justo 240 kHz tras decimar. Tres emisoras separadas más de 250 kHz no se pisan.

3. Qué añadir al informe
La figura espectro.png con las tres portadoras.
El audio demodulado de cada emisora y su espectrograma.
Una tabla de SNR por emisora, reutilizando la función audio_snr de la Parte 1 con el audio original de cada una.

Si más adelante consigues una grabación real, solo tienes que cambiar la línea de carga y el valor de fs.

Sources: SDRplay, grabaciones IQ · IQEngine como repositorio SigMF

ahora que ya he generado mi sintetica, como deberia ser mi fm_real? no lo acabo de pillar, ahora mismo es asi: #!/usr/bin/env python3
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

Con la grabación sintética lo único que cambia es cómo se carga el fichero. Además, en lo que has pegado se ha perdido la indentación en varias líneas (print, raise, return, los if). Si tu fichero está así, dará IndentationError. Te dejo el script entero y corregido.

Idea para entenderlo

La grabación es un trozo de espectro de 1,2 MHz centrado en la frecuencia del SDR, y cada emisora está a una distancia fija de ese centro. Esa distancia es --f0: en la sintética, −400 kHz, +100 kHz y +350 kHz. El script multiplica la señal por e^(−j2π·f0·n/fs), que desplaza el espectro hasta dejar la emisora elegida en 0 Hz. Luego el decimado a 240 kHz filtra las demás y el discriminador recupera el audio.

fm_real.py completo
python
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
Cómo ejecutarlo
bash
python fm_real.py fm_sintetico.cf32 --fs 1200000 --f0 0 --seconds 4

Abre figures/spectrum.png. Deberías ver tres picos, en −400 kHz, +100 kHz y +350 kHz. Luego repite para cada emisora, cambiando solo --f0. Guarda las figuras entre ejecuciones, porque cada una sobrescribe las anteriores.

--f0	Lo que deberías ver y oír
-400000	En el espectrograma, una línea que sube de 300 Hz a 3 kHz
100000	Dos líneas horizontales, a 440 y 660 Hz
350000	Un tono de 1 kHz que se enciende y se apaga dos veces por segundo
0	Ruido: no hay ninguna emisora ahí

Si con el f0 correcto el audio suena distorsionado o la línea sale mal, revisa dos cosas: que el signo de f0 es el de la tabla, y que has pasado --fs 1200000. Con otra fs, el decimado ya no deja el audio a 240 kHz.

Con --gr, la correlación con wfm_rcv puede salir menor de lo esperado, porque el bloque de GNU Radio puede usar otro desénfasis que el tuyo de 50 µs. No lo he podido comprobar aquí.