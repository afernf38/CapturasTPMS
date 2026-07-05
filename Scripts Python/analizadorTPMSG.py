#!/usr/bin/env python3
"""
Analizador TPMS 433.92 MHz – versión robusta
Pensado para señales OOK reales de sensores TPMS
"""

import numpy as np
import sys
import os
from collections import Counter

def analizar_tpms_chunks(
    archivo,
    sample_rate=2e6,
    chunk_size_seconds=10,
    min_bits=40,
    silencio_ms=5
):
    print("\n" + "="*70)
    print("ANALIZADOR TPMS 433.92 MHz (ROBUSTO)")
    print("="*70)

    bytes_per_sample = 4
    file_size = os.path.getsize(archivo)
    total_samples = file_size // bytes_per_sample
    total_duration = total_samples / sample_rate

    print(f"Archivo: {archivo}")
    print(f"Duración: {total_duration:.1f}s")
    print(f"Sample rate: {sample_rate/1e6:.1f} MHz\n")

    samples_per_chunk = int(chunk_size_seconds * sample_rate)
    silencio_samples = int((silencio_ms / 1000) * sample_rate)

    tramas_bits = []

    with open(archivo, "rb") as f:
        chunk_idx = 0
        while True:
            raw = f.read(samples_per_chunk * bytes_per_sample)
            if not raw:
                break
            chunk_idx += 1
            print(f"📦 Chunk {chunk_idx}")

            data = np.frombuffer(raw, dtype=np.int16)
            iq = (data[::2] + 1j * data[1::2]).astype(np.complex64) / 32768.0
            mag = np.abs(iq)

            # Umbral dinámico (ruido + margen)
            noise_floor = np.median(mag)
            threshold = noise_floor * 2

            above = mag > threshold

            # Transiciones
            diff = np.diff(above.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            if len(starts) == 0 or len(ends) == 0:
                continue
            if starts[0] > ends[0]:
                ends = ends[1:]
            starts = starts[:len(ends)]

            gaps = starts[1:] - ends[:-1]

            # Separar tramas por silencio largo
            cortes = np.where(gaps > silencio_samples)[0] + 1
            indices = np.split(np.arange(len(starts)), cortes)

            for idx in indices:
                if len(idx) < min_bits:
                    continue

                pulse_len = ends[idx] - starts[idx]

                # Cluster simple: corto vs largo
                med = np.median(pulse_len)
                corto = pulse_len[pulse_len < med]
                largo = pulse_len[pulse_len >= med]

                if len(corto) == 0 or len(largo) == 0:
                    continue

                thr = (np.mean(corto) + np.mean(largo)) / 2
                bits = ''.join('1' if p > thr else '0' for p in pulse_len)

                if len(bits) >= min_bits:
                    tramas_bits.append(bits)

    print("\n📊 Análisis de repetición")
    counter = Counter(tramas_bits)

    for trama, cnt in counter.most_common(10):
        if cnt < 2:
            continue
        print("-"*60)
        print(f"Repeticiones: {cnt}")
        print(f"Bits ({len(trama)}): {trama}")

        # Hex
        hex_bytes = []
        for i in range(0, len(trama)-7, 8):
            hex_bytes.append(f"{int(trama[i:i+8],2):02X}")
        print(f"Hex: {' '.join(hex_bytes)}")

    if len(counter) == 0:
        print("❌ No se detectaron tramas TPMS válidas")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("archivo")
    parser.add_argument("--sample-rate", type=float, default=2e6)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--min-bits", type=int, default=40)
    parser.add_argument("--silencio-ms", type=float, default=5)

    args = parser.parse_args()

    analizar_tpms_chunks(
        args.archivo,
        sample_rate=args.sample_rate,
        chunk_size_seconds=args.chunk_size,
        min_bits=args.min_bits,
        silencio_ms=args.silencio_ms
    )

if __name__ == "__main__":
    main()

