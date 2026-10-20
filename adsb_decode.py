#!/usr/bin/env python3
"""Decodes DF17 frames (dump1090 --raw output, or data/frames_custom.txt) with pyModeS.

Usage: python adsb_decode.py data/frames.txt
Output: data/adsb_aircraft.csv (one row per aircraft)
Uses the unified pms.decode API (pyModeS 3.x).
"""
import sys
import pandas as pd
import pyModeS as pms

path = sys.argv[1] if len(sys.argv) > 1 else 'data/frames.txt'
aircraft = {}
for line in open(path):
    msg = line.strip().strip('*;')
    if len(msg) != 28:
        continue                       # keep only long (112-bit) frames
    try:
        d = pms.decode(msg)
        if d.get('df') != 17 or not d.get('crc_valid', True) or not d.get('icao'):
            continue
    except Exception:
        continue                       # corrupted frame: skip
    a = aircraft.setdefault(d['icao'], {'ICAO': d['icao'], 'Callsign': 'Unknown',
                                        'Altitude (ft)': 'N/A', 'Speed (kt)': 'N/A'})
    if d.get('callsign'):
        a['Callsign'] = d['callsign'].strip('_ ')
    if d.get('altitude') is not None:
        a['Altitude (ft)'] = d['altitude']
    if d.get('speed') is not None:
        a['Speed (kt)'] = d['speed']

if not aircraft:
    sys.exit('No valid DF17 frames found')
table = pd.DataFrame(aircraft.values())
print(table.to_string(index=False))
table.to_csv('data/adsb_aircraft.csv', index=False)
