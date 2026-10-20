#!/usr/bin/env python3
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from gnuradio import gr, blocks, analog
from scipy.signal import correlate, welch

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
    os.makedirs('data', exist_ok=True); os.makedirs('figures', exist_ok=True)
    snr_in_list = []
    snr_out_list = []
    
    print("Calculando barrido de SNR...")
    print("SNR_In,SNR_Out")
    
    # Lista de valores de SNR corregida para el barrido
    valores_snr = [40, 30, 20, 15, 12, 10, 8, 6, 4, 2, 0]
    
    # Ejecución del barrido
    for snr in valores_snr:
        tb = FmSim(10 ** (-snr / 20))
        tb.run()
        clean, nz = np.array(tb.clean.data()), np.array(tb.nz.data())
        snr_in = 10*np.log10(np.mean(abs(clean)**2) / np.mean(abs(nz)**2))
        snr_out = audio_snr(audio, np.array(tb.out.data()))
        
        print(f"{snr_in:.1f},{snr_out:.1f}")
        snr_in_list.append(snr_in)
        snr_out_list.append(snr_out)
        
        # Guardamos la última señal limpia para el Paso 4 (usando la de SNR alta)
        if snr == 40:
            clean_signal_saved = clean

    # --- PASO 3: Guardar como CSV ---
    with open('data/fm_snr_sweep.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR_Input', 'SNR_Audio'])
        writer.writerows(zip(snr_in_list, snr_out_list))
    print("\n[✓] Archivo 'data/fm_snr_sweep.csv' guardado con éxito.")

    # --- PASO 3: Generar Gráfica del Efecto Umbral ---
    plt.figure(figsize=(8, 5))
    plt.plot(snr_in_list, snr_out_list, 'o-', color='blue', label='FM demodulation curve')
    plt.grid(True, linestyle='--')
    plt.xlabel('Input SNR (RF / channel) [dB]')
    plt.ylabel('Audio SNR (demodulated output) [dB]')
    plt.title('FM threshold effect (SNR curve)')
    plt.gca().invert_xaxis() # Invertimos para ver el decaimiento de izquierda a derecha (de 40 a 0 dB)
    plt.savefig('figures/fm_threshold.png')
    print("[✓] Gráfica 'figures/fm_threshold.png' guardada con éxito.")

    # --- PASO 4: Cálculo de la Densidad Espectral (Welch) y Regla de Carson ---
    f_axis, psd = welch(clean_signal_saved, fs=QR, nperseg=1024, return_onesided=False)
    
    # Centramos las frecuencias (FFTShift equivalente para Welch)
    f_axis = np.fft.fftshift(f_axis)
    psd = np.fft.fftshift(psd)
    
    plt.figure(figsize=(8, 4))
    plt.plot(f_axis / 1e3, 10*np.log10(psd), color='green')
    plt.grid(True, linestyle='--')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Power spectral density [dB/Hz]')
    plt.title('Clean FM signal spectrum (Carson's rule check)')
    plt.savefig('figures/fm_carson_psd.png')
    print("[✓] Gráfica de espectro 'figures/fm_carson_psd.png' guardada con éxito.")



