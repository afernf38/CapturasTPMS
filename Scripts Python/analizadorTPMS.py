#!/usr/bin/env python3
"""
Analizador TPMS Mejorado
Versión optimizada para mensajes cortos y señales débiles
"""

import numpy as np
import sys

def analizar_tpms(archivo, sample_rate=2e6, umbral=0.0001, min_pulsos=8, max_gap_ms=50):
    """
    Función principal de análisis TPMS
    
    Parámetros:
        archivo: ruta al archivo .complex16s
        sample_rate: tasa de muestreo (default 2 MHz)
        umbral: umbral de detección (default 0.0001)
        min_pulsos: mínimo de pulsos por mensaje (default 8 - MÁS REALISTA)
        max_gap_ms: gap máximo entre mensajes en ms (default 50ms)
    """
    
    print(f"\n{'='*70}")
    print(f"ANALIZADOR TPMS MEJORADO")
    print(f"{'='*70}")
    print(f"Archivo:      {archivo}")
    print(f"Sample rate:  {sample_rate/1e6:.1f} MHz")
    print(f"Umbral:       {umbral}")
    print(f"Min pulsos:   {min_pulsos}")
    print(f"Max gap:      {max_gap_ms} ms")
    print(f"{'='*70}\n")
    
    # 1. CARGAR SEÑAL
    print("📡 [1/6] Cargando señal...")
    try:
        data = np.fromfile(archivo, dtype=np.int16)
        iq_data = (data[::2] + 1j * data[1::2]).astype(np.complex64) / 32768.0
        magnitude = np.abs(iq_data)
        duracion = len(iq_data)/sample_rate
        print(f"    ✓ {len(iq_data):,} muestras")
        print(f"    ✓ Duración: {duracion:.2f} segundos")
    except Exception as e:
        print(f"    ✗ Error cargando archivo: {e}")
        return None
    
    # 2. ESTADÍSTICAS
    print(f"\n📊 [2/6] Estadísticas de la señal...")
    print(f"    Media:     {np.mean(magnitude):.6f}")
    print(f"    Máxima:    {np.max(magnitude):.6f}")
    print(f"    Mínima:    {np.min(magnitude):.6f}")
    print(f"    Desv.Std:  {np.std(magnitude):.6f}")
    
    # Percentiles
    p50 = np.percentile(magnitude, 50)
    p90 = np.percentile(magnitude, 90)
    p95 = np.percentile(magnitude, 95)
    p99 = np.percentile(magnitude, 99)
    print(f"    P50:       {p50:.6f}")
    print(f"    P90:       {p90:.6f}")
    print(f"    P95:       {p95:.6f}")
    print(f"    P99:       {p99:.6f}")
    
    # Mostrar umbrales sugeridos pero NO ajustar automáticamente
    umbral_sugerido_min = p90 * 0.5
    umbral_sugerido_max = p95 * 1.5
    print(f"\n    💡 Umbrales sugeridos: {umbral_sugerido_min:.6f} - {umbral_sugerido_max:.6f}")
    print(f"    📌 Usando umbral:      {umbral:.6f}")
    
    # 3. DETECTAR PULSOS
    print(f"\n🔍 [3/6] Detectando pulsos (umbral: {umbral:.6f})...")
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
    
    print(f"    ✓ {len(starts)} pulsos detectados")
    
    if len(starts) == 0:
        print("\n    ✗ No se detectaron pulsos.")
        print("    💡 Sugerencias:")
        print("       - Reduce el umbral (--threshold 0.00005)")
        print("       - Verifica que la señal esté en la frecuencia correcta")
        print("       - Aumenta el gain durante la captura")
        return None
    
    # Calcular duración de pulsos y gaps
    pulse_widths = ends - starts
    gaps = starts[1:] - ends[:-1]
    
    print(f"    Pulso medio:   {np.mean(pulse_widths):.1f} muestras ({np.mean(pulse_widths)/sample_rate*1e6:.2f} µs)")
    print(f"    Gap medio:     {np.mean(gaps):.1f} muestras ({np.mean(gaps)/sample_rate*1e3:.2f} ms)")
    print(f"    Gap máximo:    {np.max(gaps):.1f} muestras ({np.max(gaps)/sample_rate*1e3:.2f} ms)")
    
    # 4. AGRUPAR EN MENSAJES
    print(f"\n📦 [4/6] Agrupando en mensajes...")
    max_gap_samples = int(max_gap_ms * sample_rate / 1000)
    
    # Usar gap máximo para separar mensajes
    msg_indices = [0]
    for i, gap in enumerate(gaps):
        if gap > max_gap_samples:
            msg_indices.append(i + 1)
    msg_indices.append(len(starts))
    
    num_mensajes = len(msg_indices) - 1
    print(f"    ✓ {num_mensajes} mensajes encontrados")
    
    # Mostrar distribución de pulsos por mensaje
    pulsos_por_mensaje = []
    for i in range(len(msg_indices) - 1):
        start_idx = msg_indices[i]
        end_idx = msg_indices[i + 1]
        pulsos_por_mensaje.append(end_idx - start_idx)
    
    print(f"    Pulsos/mensaje: min={min(pulsos_por_mensaje)}, max={max(pulsos_por_mensaje)}, media={np.mean(pulsos_por_mensaje):.1f}")
    
    # 5. DECODIFICAR MENSAJES
    print(f"\n🔓 [5/6] Decodificando mensajes (mín {min_pulsos} pulsos)...")
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
            msg_pulses = msg_ends - msg_starts
            
            # Calcular tiempos
            tiempo_inicio_s = msg_starts[0] / sample_rate
            duracion_msg_ms = (msg_ends[-1] - msg_starts[0]) / sample_rate * 1000
            
            # Intentar decodificación PPM
            if len(msg_gaps) > 0:
                median_gap = np.median(msg_gaps)
                bits = ''.join(['1' if g > median_gap else '0' for g in msg_gaps])
                
                # Convertir a hex
                hex_bytes = []
                for j in range(0, len(bits) - 7, 8):
                    byte = bits[j:j+8]
                    if len(byte) == 8:
                        hex_bytes.append(f"{int(byte, 2):02X}")
                
                hex_str = ' '.join(hex_bytes)
            else:
                bits = ''
                hex_str = 'N/A'
            
            mensajes_validos.append({
                'num': i + 1,
                'pulsos': num_pulsos,
                'tiempo_s': tiempo_inicio_s,
                'duracion_ms': duracion_msg_ms,
                'hex': hex_str,
                'bits': bits,
                'pulse_mean': np.mean(msg_pulses),
                'gap_mean': np.mean(msg_gaps) if len(msg_gaps) > 0 else 0
            })
    
    print(f"    ✓ {len(mensajes_validos)} mensajes válidos decodificados")
    
    # 6. MOSTRAR RESULTADOS
    print(f"\n{'='*70}")
    print("📋 RESULTADOS DETALLADOS")
    print(f"{'='*70}\n")
    
    if len(mensajes_validos) == 0:
        print("❌ No se encontraron mensajes válidos")
        print(f"\n💡 Sugerencias:")
        print(f"   - Reduce --min-pulses (actual: {min_pulsos})")
        print(f"   - Verifica que la modulación sea correcta")
        print(f"   - Comprueba la frecuencia de captura")
        return None
    
    for msg in mensajes_validos:
        print(f"┌─ Mensaje #{msg['num']} {'─'*55}")
        print(f"│  ⏱️  Tiempo:     {msg['tiempo_s']:.3f} s")
        print(f"│  📏 Duración:   {msg['duracion_ms']:.2f} ms")
        print(f"│  📊 Pulsos:     {msg['pulsos']}")
        print(f"│  📡 Pulso med:  {msg['pulse_mean']:.1f} muestras ({msg['pulse_mean']/sample_rate*1e6:.2f} µs)")
        print(f"│  ⏸️  Gap medio:  {msg['gap_mean']:.1f} muestras ({msg['gap_mean']/sample_rate*1e6:.2f} µs)")
        print(f"│  🔢 Bits:       {msg['bits'][:80]}{'...' if len(msg['bits']) > 80 else ''}")
        print(f"│  🔐 Hex:        {msg['hex'][:60]}{'...' if len(msg['hex']) > 60 else ''}")
        print(f"└{'─'*68}\n")
    
    # 7. ANÁLISIS DE SENSORES
    print(f"{'='*70}")
    print("🎯 ANÁLISIS DE SENSORES")
    print(f"{'='*70}\n")
    
    sensores = {}
    for msg in mensajes_validos:
        if msg['hex'] != 'N/A' and len(msg['hex']) >= 11:
            # ID = primeros 4 bytes (8 caracteres hex)
            sensor_id = msg['hex'][:11].replace(' ', '')[:8]
            if sensor_id not in sensores:
                sensores[sensor_id] = []
            sensores[sensor_id].append(msg)
    
    if len(sensores) == 0:
        print("⚠️  No se pudieron identificar IDs de sensores")
    else:
        for sensor_id, msgs in sensores.items():
            print(f"🔹 Sensor ID: {sensor_id}")
            print(f"   Transmisiones: {len(msgs)}")
            print(f"   Primera:       {msgs[0]['hex'][:40]}")
            if len(msgs) > 1:
                print(f"   Última:        {msgs[-1]['hex'][:40]}")
            print()
    
    # 8. PARÁMETROS PARA URH
    if len(mensajes_validos) > 0:
        # Calcular muestras por símbolo basado en la duración media de pulso+gap
        msg_ejemplo = mensajes_validos[0]
        samples_per_symbol = int(msg_ejemplo['pulse_mean'] + msg_ejemplo['gap_mean'])
        
        print(f"{'='*70}")
        print("⚙️  PARÁMETROS SUGERIDOS PARA URH")
        print(f"{'='*70}\n")
        print(f"  Modulation:       ASK/OOK (Amplitude Shift Keying)")
        print(f"  Samples/Symbol:   {samples_per_symbol}")
        print(f"                    Alternativos: {int(samples_per_symbol/2)}, {samples_per_symbol*2}")
        print(f"  Center:           0.05 - 0.1")
        print(f"  Noise:            {umbral:.6f}")
        print(f"  Bits/Symbol:      1")
        print(f"  Pause Threshold:  {int(max_gap_samples)}")
        print()
    
    return mensajes_validos


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analizador TPMS mejorado para señales .complex16s',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  # Uso básico
  python3 analizador_tpms_mejorado.py signal.complex16s
  
  # Ajustar umbral para señales débiles
  python3 analizador_tpms_mejorado.py signal.complex16s --threshold 0.00005
  
  # Cambiar mínimo de pulsos
  python3 analizador_tpms_mejorado.py signal.complex16s --min-pulses 5
  
  # Cambiar sample rate
  python3 analizador_tpms_mejorado.py signal.complex16s --sample-rate 250000
        '''
    )
    
    parser.add_argument('archivo', help='Archivo .complex16s a analizar')
    parser.add_argument('--sample-rate', '-s', type=float, default=2e6,
                        help='Sample rate en Hz (default: 2000000)')
    parser.add_argument('--threshold', '-t', type=float, default=0.0001,
                        help='Umbral de detección (default: 0.0001)')
    parser.add_argument('--min-pulses', '-m', type=int, default=8,
                        help='Mínimo de pulsos por mensaje (default: 8)')
    parser.add_argument('--max-gap', '-g', type=float, default=50,
                        help='Gap máximo entre mensajes en ms (default: 50)')
    
    args = parser.parse_args()
    
    resultados = analizar_tpms(
        args.archivo,
        sample_rate=args.sample_rate,
        umbral=args.threshold,
        min_pulsos=args.min_pulses,
        max_gap_ms=args.max_gap
    )
    
    if resultados is None:
        sys.exit(1)

if __name__ == '__main__':
    main()
