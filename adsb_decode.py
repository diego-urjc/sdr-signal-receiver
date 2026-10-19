#!/usr/bin/env python3
"""Decodes DF17 frames from frames.txt (dump1090 --raw output) with pyModeS 2.x.

Usage: python adsb_decode.py data/frames.txt
Output: data/adsb_aircraft.csv (one row per aircraft)
"""
import sys
import pandas as pd
import pyModeS as pms

path = sys.argv[1] if len(sys.argv) > 1 else 'data/frames.txt'
rows = []
for line in open(path):
    msg = line.strip().strip('*;')
    if len(msg) != 28 or pms.df(msg) != 17 or pms.crc(msg) != 0:
        continue
    tc = pms.adsb.typecode(msg)
    row = {'icao': pms.adsb.icao(msg), 'tc': tc}
    if 1 <= tc <= 4:
        row['callsign'] = pms.adsb.callsign(msg).strip('_')
    elif 9 <= tc <= 18:
        row['alt_ft'] = pms.adsb.altitude(msg)
    elif tc == 19:
        v = pms.adsb.velocity(msg)          # (speed kt, heading deg, vrate ft/min, type)
        row['speed_kt'], row['heading_deg'], row['vrate_fpm'] = v[0], v[1], v[2]
    rows.append(row)

df = pd.DataFrame(rows)
print(f"{len(df)} valid DF17 messages")
if df.empty:
    sys.exit()
cols = [c for c in ('callsign', 'alt_ft', 'speed_kt', 'heading_deg') if c in df]
table = df.groupby('icao').agg({c: 'last' for c in cols}).join(df.groupby('icao').size().rename('n_msgs'))
print(table.to_string())
table.to_csv('data/adsb_aircraft.csv')
