#!/usr/bin/env python3
"""
Generador de Informe TPMS
Analiza archivo y genera informe detallado con todas las tramas
"""

import numpy as np
import sys
import os
from datetime import datetime

def analizar_y_generar_informe(archivo, sample_rate=2e6, umbral=0.0001, min_pulsos=50, 
                                chunk_size_seconds=30, output_file=None):
    """
    Analiza archivo TPMS y genera informe detallado
    """
    
    # Nombre del informe
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"informe_tpms_{timestamp}.txt"
    
    print(f"\n{'='*70}")
    print(f"GENERADOR DE INFORME TPMS")
    print(f"{'='*70}")
    print(f"Archivo entrada:  {archivo}")
    print(f"Archivo salida:   {output_file}")
    print(f"{'='*70}\n")
    
    # Obtener tamaño del archivo y calcular duración real
    file_size = os.path.getsize(archivo)
    bytes_per_sample = 4  # 2 bytes I + 2 bytes Q (int16)
    total_samples = file_size // bytes_per_sample
    total_duration = total_samples / sample_rate
    
    print(f"📊 Información del archivo:")
    print(f"   Tamaño:     {file_size / (1024**2):.2f} MB")
    print(f"   Samples:    {total_samples:,}")
    print(f"   Duración:   {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print(f"   Sample rate: {sample_rate/1e6:.1f} MHz")
    print()
    
    # Calcular chunks
    samples_per_chunk = int(chunk_size_seconds * sample_rate)
    num_chunks = int(np.ceil(total_samples / samples_per_chunk))
    
    print(f"🔧 Procesamiento:")
    print(f"   Chunks:     {num_chunks}")
    print(f"   Chunk size: {chunk_size_seconds}s")
    print()
    
    # Procesar archivo por chunks
    mensajes_totales = []
    
    with open(archivo, 'rb') as f:
        for chunk_idx in range(num_chunks):
            print(f"📦 Procesando chunk {chunk_idx + 1}/{num_chunks}...", end=' ')
            
            # Leer chunk
            chunk_bytes = samples_per_chunk * bytes_per_sample
            raw_data = f.read(chunk_bytes)
            
            if len(raw_data) == 0:
                print("vacío")
                break
            
            # Convertir a IQ
            data = np.frombuffer(raw_data, dtype=np.int16)
            iq_data = (data[::2] + 1j * data[1::2]).astype(np.complex64) / 32768.0
            magnitude = np.abs(iq_data)
            
            # Detectar pulsos
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
            
            if len(starts) == 0:
                print("sin pulsos")
                continue
            
            print(f"{len(starts)} pulsos →", end=' ')
            
            # Agrupar en mensajes
            gaps = starts[1:] - ends[:-1]
            if len(gaps) == 0:
                print("sin mensajes")
                continue
            
            gap_threshold = np.percentile(gaps, 75)
            
            msg_indices = [0]
            for i, gap in enumerate(gaps):
                if gap > gap_threshold:
                    msg_indices.append(i + 1)
            msg_indices.append(len(starts))
            
            # Decodificar mensajes
            chunk_mensajes = 0
            for i in range(len(msg_indices) - 1):
                start_idx = msg_indices[i]
                end_idx = msg_indices[i + 1]
                num_pulsos = end_idx - start_idx
                
                if num_pulsos >= min_pulsos:
                    # Calcular tiempo absoluto en ms
                    tiempo_absoluto_ms = (chunk_idx * samples_per_chunk + starts[start_idx]) / sample_rate * 1000
                    
                    # Extraer pulsos del mensaje
                    msg_starts = starts[start_idx:end_idx]
                    msg_ends = ends[start_idx:end_idx]
                    msg_gaps = msg_starts[1:] - msg_ends[:-1]
                    msg_pulses = msg_ends - msg_starts
                    
                    # Decodificar - intentar múltiples métodos
                    if len(msg_gaps) > 0:
                        # Método 1: PWM (basado en duración del pulso)
                        median_pulse = np.median(msg_pulses)
                        bits_pwm = ''.join(['1' if p > median_pulse else '0' for p in msg_pulses])
                        
                        # Método 2: PPM (basado en gaps)
                        median_gap = np.median(msg_gaps)
                        bits_ppm = ''.join(['1' if g > median_gap else '0' for g in msg_gaps])
                        
                        # Elegir el método con mejor balance
                        ones_pwm = bits_pwm.count('1')
                        ones_ppm = bits_ppm.count('1')
                        
                        ratio_pwm = abs(ones_pwm / len(bits_pwm) - 0.5) if len(bits_pwm) > 0 else 1
                        ratio_ppm = abs(ones_ppm / len(bits_ppm) - 0.5) if len(bits_ppm) > 0 else 1
                        
                        if ratio_pwm < ratio_ppm:
                            bits = bits_pwm
                            metodo = "PWM"
                        else:
                            bits = bits_ppm
                            metodo = "PPM"
                        
                        # Convertir a hex
                        hex_bytes = []
                        for j in range(0, len(bits) - 7, 8):
                            byte = bits[j:j+8]
                            if len(byte) == 8:
                                hex_bytes.append(f"{int(byte, 2):02X}")
                        
                        hex_str = ' '.join(hex_bytes)
                        
                        # Formatear bits en bloques de 64
                        bits_formatted = []
                        for j in range(0, len(bits), 64):
                            bits_formatted.append(bits[j:j+64])
                        
                    else:
                        bits = ''
                        hex_str = 'N/A'
                        metodo = 'N/A'
                        bits_formatted = []
                    
                    mensajes_totales.append({
                        'num': len(mensajes_totales) + 1,
                        'tiempo_ms': tiempo_absoluto_ms,
                        'pulsos': num_pulsos,
                        'bits': bits,
                        'bits_formatted': bits_formatted,
                        'hex': hex_str,
                        'metodo': metodo
                    })
                    
                    chunk_mensajes += 1
            
            print(f"{chunk_mensajes} mensajes")
    
    # Generar informe
    print(f"\n{'='*70}")
    print(f"📝 GENERANDO INFORME...")
    print(f"{'='*70}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Encabezado
        f.write("=" * 80 + "\n")
        f.write("RESULTADOS DE ANÁLISIS TPMS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Archivo analizado: {os.path.basename(archivo)}\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write(f"Duración: {total_duration:.2f} segundos\n")
        f.write(f"Mensajes decodificados: {len(mensajes_totales)}\n")
        f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Resumen de sensores
        sensores = {}
        for msg in mensajes_totales:
            if msg['hex'] != 'N/A' and len(msg['hex']) >= 11:
                sensor_id = msg['hex'][:11].replace(' ', '')[:8]
                if sensor_id not in sensores:
                    sensores[sensor_id] = []
                sensores[sensor_id].append(msg)
        
        if len(sensores) > 0:
            f.write("=" * 80 + "\n")
            f.write("RESUMEN DE SENSORES\n")
            f.write("=" * 80 + "\n\n")
            
            for sensor_id, msgs in sensores.items():
                f.write(f"Sensor ID: {sensor_id}\n")
                f.write(f"  Transmisiones: {len(msgs)}\n")
                
                if len(msgs) > 1:
                    intervalos = []
                    for i in range(1, len(msgs)):
                        intervalo = (msgs[i]['tiempo_ms'] - msgs[i-1]['tiempo_ms']) / 1000
                        intervalos.append(intervalo)
                    
                    f.write(f"  Intervalo medio: {np.mean(intervalos):.2f}s\n")
                    f.write(f"  Intervalo min: {np.min(intervalos):.2f}s\n")
                    f.write(f"  Intervalo max: {np.max(intervalos):.2f}s\n")
                
                f.write("\n")
        
        # Detalles de cada mensaje
        f.write("\n" + "=" * 80 + "\n")
        f.write("MENSAJES DETALLADOS\n")
        f.write("=" * 80 + "\n\n")
        
        for msg in mensajes_totales:
            f.write("=" * 80 + "\n")
            f.write(f"MENSAJE #{msg['num']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Tiempo:  {msg['tiempo_ms']:.2f} ms ({msg['tiempo_ms']/1000:.2f} s)\n")
            f.write(f"Pulsos:  {msg['pulsos']}\n")
            f.write(f"Bits:    {len(msg['bits'])}\n")
            f.write(f"Método:  {msg['metodo']}\n")
            f.write(f"\nHex:     {msg['hex']}\n")
            f.write(f"\nBinario:\n")
            
            for bits_line in msg['bits_formatted']:
                f.write(f"  {bits_line}\n")
            
            # Si quedan bits sin formatear
            if len(msg['bits']) % 64 != 0:
                remaining = msg['bits'][-(len(msg['bits']) % 64):]
                if remaining and remaining not in msg['bits_formatted'][-1] if msg['bits_formatted'] else True:
                    f.write(f"  {remaining}\n")
            
            f.write("\n")
    
    print(f"✅ Informe generado: {output_file}")
    print(f"   Total mensajes: {len(mensajes_totales)}")
    print(f"   Total sensores: {len(sensores)}")
    print()
    
    return output_file, mensajes_totales


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Genera informe detallado de análisis TPMS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  # Generar informe básico
  python3 generar_informe_tpms.py archivo.complex16s
  
  # Con parámetros personalizados
  python3 generar_informe_tpms.py archivo.complex16s --threshold 0.0001 --min-pulses 50
  
  # Especificar nombre de salida
  python3 generar_informe_tpms.py archivo.complex16s -o mi_informe.txt
        '''
    )
    
    parser.add_argument('archivo', help='Archivo .complex16s a analizar')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Archivo de salida (default: informe_tpms_TIMESTAMP.txt)')
    parser.add_argument('--sample-rate', '-s', type=float, default=2e6,
                        help='Sample rate en Hz (default: 2000000)')
    parser.add_argument('--threshold', '-t', type=float, default=0.0001,
                        help='Umbral de detección (default: 0.0001)')
    parser.add_argument('--min-pulses', '-m', type=int, default=50,
                        help='Mínimo de pulsos por mensaje (default: 50)')
    parser.add_argument('--chunk-size', '-c', type=int, default=30,
                        help='Tamaño de chunk en segundos (default: 30)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.archivo):
        print(f"❌ Error: Archivo no encontrado: {args.archivo}")
        sys.exit(1)
    
    informe, mensajes = analizar_y_generar_informe(
        args.archivo,
        sample_rate=args.sample_rate,
        umbral=args.threshold,
        min_pulsos=args.min_pulses,
        chunk_size_seconds=args.chunk_size,
        output_file=args.output
    )
    
    print(f"📄 Para ver el informe:")
    print(f"   cat {informe}")
    print(f"   # o")
    print(f"   less {informe}")
    print()

if __name__ == '__main__':
    main()
