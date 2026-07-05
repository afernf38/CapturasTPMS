#!/usr/bin/env python3
"""
Captura TPMS en Tiempo Real
Detecta automáticamente transmisiones TPMS y guarda solo las ráfagas válidas
"""

import numpy as np
import subprocess
import signal
import sys
import time
from datetime import datetime
import os

class TPMSCapture:
    def __init__(self, frequency=433.92e6, sample_rate=2e6, gain=40, threshold=0.01):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self.threshold = threshold
        self.capturing = False
        self.process = None
        
    def detect_signal(self, data, min_duration_ms=10):
        """Detecta si hay señal TPMS en los datos"""
        magnitude = np.abs(data)
        
        # Buscar picos significativos
        above_threshold = magnitude > self.threshold
        
        if not np.any(above_threshold):
            return False, 0
        
        # Calcular duración de la señal
        transitions = np.diff(above_threshold.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        if len(starts) > 0 and len(ends) > 0:
            # Ajustar pares
            if starts[0] > ends[0]:
                ends = ends[1:]
            if len(starts) > len(ends):
                starts = starts[:len(ends)]
            
            if len(starts) > 0:
                total_duration_ms = (ends[-1] - starts[0]) / self.sample_rate * 1000
                num_pulses = len(starts)
                
                # TPMS típico: 5-100ms de duración, 10-100 pulsos
                if total_duration_ms >= min_duration_ms and num_pulses >= 5:
                    return True, num_pulses
        
        return False, 0
    
    def capture_continuous(self, duration_seconds=60, output_dir="tpms_captures"):
        """Captura continua buscando señales TPMS"""
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"🎯 CAPTURA TPMS EN TIEMPO REAL")
        print(f"{'='*70}")
        print(f"Frecuencia:    {self.frequency/1e6:.2f} MHz")
        print(f"Sample rate:   {self.sample_rate/1e6:.1f} MHz")
        print(f"Gain:          {self.gain} dB")
        print(f"Umbral:        {self.threshold}")
        print(f"Duración:      {duration_seconds} segundos")
        print(f"Directorio:    {output_dir}/")
        print(f"{'='*70}\n")
        
        print("⏳ Iniciando captura HackRF...")
        print("💡 Activa los sensores TPMS (conduce, deflacta rueda, etc.)\n")
        
        # Buffer para almacenar datos
        chunk_size = int(self.sample_rate * 0.5)  # 500ms chunks
        detections = 0
        total_chunks = 0
        
        # Comando hackrf_transfer
        cmd = [
            'hackrf_transfer',
            '-r', '/dev/stdout',
            '-f', str(int(self.frequency)),
            '-s', str(int(self.sample_rate)),
            '-g', str(self.gain),
            '-l', str(self.gain),
            '-a', '1'  # Amp enable
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=chunk_size * 4
            )
            
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                # Leer chunk
                raw_data = self.process.stdout.read(chunk_size * 4)
                
                if len(raw_data) < chunk_size * 4:
                    break
                
                # Convertir a IQ
                data = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32)
                iq_data = (data[::2] + 1j * data[1::2]) / 128.0
                
                total_chunks += 1
                
                # Detectar señal
                is_signal, num_pulses = self.detect_signal(iq_data)
                
                if is_signal:
                    detections += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{output_dir}/tpms_{timestamp}_p{num_pulses}.complex64"
                    
                    # Guardar como complex64
                    iq_data.astype(np.complex64).tofile(filename)
                    
                    print(f"✓ [{detections}] Señal detectada: {num_pulses} pulsos → {filename}")
                
                # Mostrar progreso cada 5 segundos
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and total_chunks % 10 == 0:
                    print(f"⏱️  {elapsed:.0f}s transcurridos | {detections} detecciones")
            
            print(f"\n{'='*70}")
            print(f"✓ Captura completada")
            print(f"  Tiempo:       {elapsed:.1f} segundos")
            print(f"  Chunks:       {total_chunks}")
            print(f"  Detecciones:  {detections}")
            print(f"{'='*70}\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Captura interrumpida por el usuario")
        
        finally:
            if self.process:
                self.process.terminate()
                self.process.wait()
    
    def scan_frequencies(self, freq_list_mhz, duration_per_freq=10):
        """Escanea múltiples frecuencias buscando TPMS"""
        
        print(f"\n{'='*70}")
        print(f"🔍 ESCANEO DE FRECUENCIAS TPMS")
        print(f"{'='*70}\n")
        
        results = {}
        
        for freq_mhz in freq_list_mhz:
            self.frequency = freq_mhz * 1e6
            print(f"\n📡 Escaneando {freq_mhz} MHz ({duration_per_freq}s)...")
            
            output_dir = f"scan_{freq_mhz}MHz"
            self.capture_continuous(duration_per_freq, output_dir)
            
            # Contar detecciones
            if os.path.exists(output_dir):
                files = [f for f in os.listdir(output_dir) if f.endswith('.complex64')]
                results[freq_mhz] = len(files)
                print(f"   → {len(files)} señales detectadas")
            else:
                results[freq_mhz] = 0
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DEL ESCANEO")
        print(f"{'='*70}\n")
        
        for freq, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * min(count, 50)
            print(f"  {freq:7.2f} MHz: {bar} ({count} señales)")
        
        if max(results.values()) > 0:
            best_freq = max(results.items(), key=lambda x: x[1])
            print(f"\n🎯 Mejor frecuencia: {best_freq[0]} MHz ({best_freq[1]} señales)")
        else:
            print(f"\n❌ No se detectaron señales en ninguna frecuencia")
            print(f"💡 Sugerencias:")
            print(f"   - Aumenta el gain (actual: {self.gain})")
            print(f"   - Reduce el umbral (actual: {self.threshold})")
            print(f"   - Verifica que los sensores TPMS estén activos")
        
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Captura TPMS en tiempo real con detección automática',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  # Captura básica en 433.92 MHz
  sudo python3 captura_tpms_realtime.py
  
  # Captura con parámetros personalizados
  sudo python3 captura_tpms_realtime.py -f 434.42 -g 40 -t 0.005 -d 120
  
  # Escanear múltiples frecuencias
  sudo python3 captura_tpms_realtime.py --scan 433.92 434.42 315.0
  
  # Captura larga con alta ganancia
  sudo python3 captura_tpms_realtime.py -g 50 -d 300
        '''
    )
    
    parser.add_argument('-f', '--frequency', type=float, default=433.92,
                        help='Frecuencia en MHz (default: 433.92)')
    parser.add_argument('-s', '--sample-rate', type=float, default=2.0,
                        help='Sample rate en MHz (default: 2.0)')
    parser.add_argument('-g', '--gain', type=int, default=40,
                        help='Ganancia en dB (default: 40)')
    parser.add_argument('-t', '--threshold', type=float, default=0.01,
                        help='Umbral de detección (default: 0.01)')
    parser.add_argument('-d', '--duration', type=int, default=60,
                        help='Duración de captura en segundos (default: 60)')
    parser.add_argument('-o', '--output', type=str, default='tpms_captures',
                        help='Directorio de salida (default: tpms_captures)')
    parser.add_argument('--scan', nargs='+', type=float,
                        help='Escanear múltiples frecuencias (en MHz)')
    
    args = parser.parse_args()
    
    # Verificar que hackrf_transfer esté disponible
    try:
        subprocess.run(['hackrf_transfer', '--help'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: hackrf_transfer no encontrado")
        print("   Instala con: sudo apt-get install hackrf")
        sys.exit(1)
    
    # Crear capturador
    capturer = TPMSCapture(
        frequency=args.frequency * 1e6,
        sample_rate=args.sample_rate * 1e6,
        gain=args.gain,
        threshold=args.threshold
    )
    
    # Modo escaneo o captura normal
    if args.scan:
        capturer.scan_frequencies(args.scan, duration_per_freq=args.duration)
    else:
        capturer.capture_continuous(args.duration, args.output)


if __name__ == '__main__':
    main()
