#!/usr/bin/env python3
"""Custom ADS-B demodulator (challenge) and comparison with dump1090.

Usage: python adsb_demod.py data/modes1.bin data/frames.txt
uint8 IQ at 2 Msps: 1 bit = 2 samples, preamble = 16 samples (pulses at 0, 2, 7, 9).
"""
import os
import sys
import numpy as np

CRC_POLY = 0x1FFF409


def crc24(bits):                      # bits: 112 ints 0/1; 0 = valid frame
    reg = 0
    for b in bits:
        reg = (reg << 1) | int(b)
        if reg & (1 << 24):
            reg ^= CRC_POLY
    return reg


def bits_to_hex(bits):
    return '%028x' % int(''.join(map(str, bits)), 2)


def demodulate(path):
    raw = np.fromfile(path, np.uint8)
    iq = (raw[0::2] - 127.5) + 1j * (raw[1::2] - 127.5)
    mag = np.abs(iq)
    n = len(mag) - (16 + 224)
    i = np.arange(n)
    # candidates: preamble shape (pulses at 0,2,7,9 above their neighbors)
    pulses = mag[i] + mag[i+2] + mag[i+7] + mag[i+9]
    cand = ((mag[i] > mag[i+1]) & (mag[i+1] < mag[i+2]) & (mag[i+2] > mag[i+3]) &
            (mag[i+3] < mag[i]) & (mag[i+4] < mag[i]) & (mag[i+5] < mag[i]) &
            (mag[i+6] < mag[i]) & (mag[i+7] > mag[i+8]) & (mag[i+8] < mag[i+9]) &
            (mag[i+9] > mag[i+6]))
    frames, skip_until = [], -1
    for p in np.flatnonzero(cand):
        if p < skip_until:
            continue
        d = mag[p+16:p+16+224]
        bits = (d[0::2] > d[1::2]).astype(np.uint8)
        if crc24(bits) == 0:
            frames.append(bits_to_hex(bits))
            skip_until = p + 16 + 224          # avoid decoding the same frame twice
    return frames


def load_reference(path):
    out = []
    for line in open(path):
        m = line.strip().strip('*;')
        if len(m) == 28:
            out.append(m.lower())
    return out


if __name__ == '__main__':
    binf = sys.argv[1] if len(sys.argv) > 1 else 'data/modes1.bin'
    ref = sys.argv[2] if len(sys.argv) > 2 else 'data/frames.txt'
    mine = demodulate(binf)
    with open('data/frames_custom.txt', 'w') as f:
        f.writelines(f'*{m};\n' for m in mine)
    if not os.path.exists(ref):
        print(f"Custom:    {len(mine)} frames ({len(set(mine))} unique) -> data/frames_custom.txt")
        sys.exit(f"{ref} not found: run dump1090 first to compare")
    # DF17/18 only (CRC=0 without applying the address mask); dump1090 also returns other DFs
    ref_all = load_reference(ref)
    ref_df17 = [m for m in ref_all if (int(m[:2], 16) >> 3) in (17, 18)]
    s_mine, s_ref = set(mine), set(ref_df17)
    print(f"Custom:    {len(mine)} frames ({len(s_mine)} unique)")
    print(f"dump1090:  {len(ref_df17)} DF17/18 frames ({len(s_ref)} unique, of {len(ref_all)} total)")
    print(f"Matching:  {len(s_mine & s_ref)}")
    print(f"Custom only: {len(s_mine - s_ref)}   dump1090 only: {len(s_ref - s_mine)}")
