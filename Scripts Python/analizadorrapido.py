#!/usr/bin/env python3
"""
Análisis rápido de señales TPMS
Script simplificado para uso directo
"""

import numpy as np
import sys

def analizar_tpms(archivo, sample_rate=2e6, umbral=0.0001, min_pulsos=50):
    """
    Función principal de análisis TPMS
    
    Parámetros:
        archivo: ruta al archivo .complex16s
        sample_rate: tasa de muestreo (default 2 MHz)
        umbral: umbral de detección (default 0.0001)
        min_pulsos: mínimo de pulsos por mensaje (default 50)
    """
    
    print(f"\n{'='*60}")
    print(f"Analizando: {archivo}")
    print(f"{'='*60}\n")
    
    # 1. CARGAR SEÑAL
    print("1. Cargando señal...")
    data = np.fromfile(archivo, dtype=np.int16)
    iq_data = (data[::2] + 1j * data[1::2]).astype(np.complex64) / 32768.0
    magnitude = np.abs(iq_data)
    print(f"   ✓ {len(iq_data)} muestras, duración: {len(iq_data)/sample_rate:.2f}s")
    
    # 2. DETECTAR PULSOS
    print("\n2. Detectando pulsos...")
    above = magnitude > umbral
    transitions = np.diff(above.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Ajustar pares
    if len(starts) > 0 and len(ends) > 0:
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:len(ends)]
    
    print(f"   ✓ {len(starts)} pulsos detectados")
    
    if len(starts) == 0:
        print("\n   ✗ No se detectaron pulsos. Ajusta el umbral.")
        return
    
    # 3. AGRUPAR EN MENSAJES
    print("\n3. Agrupando en mensajes...")
    gaps = starts[1:] - ends[:-1]
    gap_threshold = np.percentile(gaps, 75)
    
    msg_indices = [0]
    for i, gap in enumerate(gaps):
        if gap > gap_threshold:
            msg_indices.append(i + 1)
    msg_indices.append(len(starts))
    
    print(f"   ✓ {len(msg_indices)-1} mensajes encontrados")
    
    # 4. DECODIFICAR MENSAJES
    print("\n4. Decodificando mensajes válidos...")
    mensajes_validos = []
    
    for i in range(len(msg_indices) - 1):
        start_idx = msg_indices[i]
        end_idx = msg_indices[i + 1]
        num_pulsos = end_idx - start_idx
        
        if num_pulsos >= min_pulsos:
            # Extraer pulsos del mensaje
            msg_starts = starts[start_idx:end_idx]
            msg_ends = ends[start_idx:end_idx]
            msg_gaps = msg_starts[1:] - msg_ends[:-1]
            
            # Decodificar PPM (gap largo=1, corto=0)
            median_gap = np.median(msg_gaps)
            bits = ''.join(['1' if g > median_gap else '0' for g in msg_gaps])
            
            # Convertir a hex
            hex_bytes = []
            for j in range(0, len(bits)-7, 8):
                byte = bits[j:j+8]
                hex_bytes.append(f"{int(byte, 2):02X}")
            
            mensajes_validos.append({
                'num': i+1,
                'pulsos': num_pulsos,
                'tiempo_ms': msg_starts[0] / sample_rate * 1000,
                'hex': ' '.join(hex_bytes),
                'bits': bits
            })
    
    print(f"   ✓ {len(mensajes_validos)} mensajes válidos (≥{min_pulsos} pulsos)")
    
    # 5. MOSTRAR RESULTADOS
    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}\n")
    
    for msg in mensajes_validos[:10]:  # Mostrar primeros 10
        print(f"Mensaje #{msg['num']}:")
        print(f"  Tiempo: {msg['tiempo_ms']:.2f} ms")
        print(f"  Pulsos: {msg['pulsos']}")
        print(f"  Hex:    {msg['hex']}")
        print()
    
    if len(mensajes_validos) > 10:
        print(f"... y {len(mensajes_validos)-10} mensajes más\n")
    
    # 6. ANALIZAR SENSORES
    print("Análisis de sensores:")
    sensores = {}
    for msg in mensajes_validos:
        # ID = primeros 4 bytes
        sensor_id = ' '.join(msg['hex'].split()[:4])
        if sensor_id not in sensores:
            sensores[sensor_id] = []
        sensores[sensor_id].append(msg['hex'])
    
    for sensor_id, transmisiones in sensores.items():
        print(f"\n  Sensor ID: {sensor_id}")
        print(f"  Transmisiones: {len(transmisiones)}")
        if len(transmisiones) > 0:
            print(f"  Ejemplo: {transmisiones[0]}")
    
    # 7. PARÁMETROS PARA URH
    median_gap = np.median(gaps)
    median_pulse = np.median(ends - starts)
    samples_per_symbol = int(median_pulse + median_gap)
    
    print(f"\n{'='*60}")
    print("PARÁMETROS PARA URH")
    print(f"{'='*60}")
    print(f"\nModulation:       ASK/OOK")
    print(f"Samples/Symbol:   {samples_per_symbol}")
    print(f"                  (o prueba: {int(median_gap)}, {samples_per_symbol*2})")
    print(f"Center:           0.05")
    print(f"Noise:            0.0001")
    print(f"Bits/Symbol:      1\n")
    
    return mensajes_validos


# EJEMPLO DE USO
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nUso: python3 analisis_rapido.py archivo.complex16s")
        print("\nEjemplo:")
        print("  python3 analisis_rapido.py señal.complex16s")
        print("\nOpcional - cambiar parámetros:")
        print("  # En Python:")
        print("  mensajes = analizar_tpms('señal.complex16s', umbral=0.0005)")
        sys.exit(1)
    
    archivo = sys.argv[1]
    resultados = analizar_tpms(archivo)
