# Software-defined radio signal reception and analysis

Software-defined radio simulation and analysis with GNU Radio and Python: an FM chain with an SNR sweep,
demodulation of real IQ recordings, and ADS-B decoding with a custom demodulator.

## Contents

| File | Description |
|---|---|
| `fm_sim.py` | Simulated WBFM chain (GNU Radio), SNR sweep, CSV, plots and PSD (Carson's rule) |
| `fm_real.py` | Custom FM demodulator for a real IQ recording; spectrum, audio and spectrogram |
| `adsb_decode.py` | Decodes dump1090 DF17 frames with pyModeS and builds a per-aircraft table |
| `adsb_demod.py` | Custom ADS-B demodulator (PPM at 2 Msps, CRC-24) and comparison with dump1090 |
| `flowgraphs/` | GNU Radio Companion `.grc` flowgraphs |
| `data/` | CSVs and results (third-party files are not included) |
| `figures/` | Report plots |
| `report.pdf` | Technical report |

## Installation (Ubuntu)

```bash
sudo apt install -y gnuradio python3-numpy python3-scipy python3-matplotlib python3-pandas \
  build-essential git librtlsdr-dev libusb-1.0-0-dev pkg-config
python3 -m venv --system-site-packages venv && source venv/bin/activate
pip install pyModeS
python3 -c "from gnuradio import analog; print('ok')"
```

## Usage

```bash
python fm_sim.py                                   # -> data/fm_snr_sweep.csv, figures/fm_*.png
python fm_real.py data/fm_rds.wav --f0 0 [--gr]    # -> figures/spectrum*.png, data/fm_demod.wav
git clone https://github.com/antirez/dump1090.git && (cd dump1090 && make)
./dump1090/dump1090 --ifile dump1090/testfiles/modes1.bin --raw > data/frames.txt
cp dump1090/testfiles/modes1.bin data/             # local only, not uploaded
python adsb_decode.py data/frames.txt              # -> data/adsb_aircraft.csv
python adsb_demod.py data/modes1.bin data/frames.txt
```

## Results

_(To be completed: SNR sweep plot, station spectrum, aircraft table, demodulator comparison.)_

## Limitations

- Simulation and third-party recordings: there is no capture of our own with hardware.
- Ideal AWGN noise; no fading or multipath in the sweep.
- No real clock synchronization (no oscillator drift or frequency offset).
- Third-party files (SDRplay recording, dump1090's `modes1.bin`) have their own licenses: they are not included in the repo, only linked.

## References

- GNU Radio: https://www.gnuradio.org
- SDRplay, sample IQ recordings: https://www.sdrplay.com
- dump1090 (antirez): https://github.com/antirez/dump1090
- pyModeS: https://mode-s.org
- SigMF (IQ metadata format): https://sigmf.org
