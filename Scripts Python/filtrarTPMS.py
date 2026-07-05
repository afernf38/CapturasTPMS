#!/usr/bin/env python3
"""
Filtrador y Agrupador de Tramas TPMS
Analiza un informe TPMS y agrupa las tramas por sensor, descartando ruido
"""

import re
import sys
from collections import defaultdict
import numpy as np

def parsear_informe(archivo_informe):
    """
    Lee el informe TPMS y extrae todos los mensajes
    """
    
    mensajes = []
    
    with open(archivo_informe, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    # Dividir por mensajes
    bloques = re.split(r'={80}\nMENSAJE #(\d+)', contenido)
    
    # El primer bloque es el encabezado, lo saltamos
    for i in range(1, len(bloques), 2):
        if i+1 >= len(bloques):
            break
            
        num_mensaje = int(bloques[i])
        datos = bloques[i+1]
        
        # Extraer información
        tiempo_match = re.search(r'Tiempo:\s+([\d.]+)\s+ms', datos)
        pulsos_match = re.search(r'Pulsos:\s+(\d+)', datos)
        bits_match = re.search(r'Bits:\s+(\d+)', datos)
        metodo_match = re.search(r'Método:\s+(\w+)', datos)
        hex_match = re.search(r'Hex:\s+([0-9A-F\s]+)', datos)
        
        # Extraer binario
        binario_match = re.search(r'Binario:\n((?:\s+[01]+\n?)+)', datos)
        bits_str = ''
        if binario_match:
            lineas_bits = binario_match.group(1).strip().split('\n')
            bits_str = ''.join([linea.strip() for linea in lineas_bits])
        
        if tiempo_match and hex_match:
            mensaje = {
                'num': num_mensaje,
                'tiempo_ms': float(tiempo_match.group(1)),
                'tiempo_s': float(tiempo_match.group(1)) / 1000,
                'pulsos': int(pulsos_match.group(1)) if pulsos_match else 0,
                'num_bits': int(bits_match.group(1)) if bits_match else 0,
                'metodo': metodo_match.group(1) if metodo_match else 'N/A',
                'hex': hex_match.group(1).strip(),
                'bits': bits_str
            }
            mensajes.append(mensaje)
    
    return mensajes


def extraer_id_sensor(hex_str):
    """
    Extrae el ID del sensor (primeros 4 bytes = 8 caracteres hex)
    """
    hex_clean = hex_str.replace(' ', '')
    if len(hex_clean) >= 8:
        return hex_clean[:8]
    return None


def calcular_similitud(hex1, hex2):
    """
    Calcula similitud entre dos tramas (porcentaje de bytes iguales)
    """
    bytes1 = hex1.replace(' ', '')
    bytes2 = hex2.replace(' ', '')
    
    if len(bytes1) != len(bytes2):
        return 0.0
    
    matches = sum(1 for a, b in zip(bytes1, bytes2) if a == b)
    return matches / len(bytes1)


def agrupar_mensajes(mensajes, min_mensajes=3, max_variacion_longitud=10):
    """
    Agrupa mensajes por sensor identificando patrones repetitivos
    
    Args:
        mensajes: Lista de mensajes parseados
        min_mensajes: Mínimo de mensajes para considerar un grupo válido
        max_variacion_longitud: Variación máxima permitida en longitud de trama
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 AGRUPANDO MENSAJES POR SENSOR")
    print(f"{'='*70}\n")
    
    # Agrupar por ID de sensor (primeros 4 bytes)
    grupos_por_id = defaultdict(list)
    
    for msg in mensajes:
        sensor_id = extraer_id_sensor(msg['hex'])
        if sensor_id:
            grupos_por_id[sensor_id].append(msg)
    
    print(f"📊 Agrupación por ID de sensor:")
    print(f"   Total IDs únicos: {len(grupos_por_id)}")
    print()
    
    # Filtrar grupos con suficientes mensajes
    grupos_validos = {}
    grupos_descartados = {}
    
    for sensor_id, msgs in grupos_por_id.items():
        if len(msgs) >= min_mensajes:
            grupos_validos[sensor_id] = msgs
        else:
            grupos_descartados[sensor_id] = msgs
    
    print(f"✅ Grupos válidos (≥{min_mensajes} mensajes): {len(grupos_validos)}")
    print(f"❌ Grupos descartados (<{min_mensajes} mensajes): {len(grupos_descartados)}")
    print()
    
    # Analizar cada grupo válido
    sensores_tpms = []
    
    for sensor_id, msgs in grupos_validos.items():
        # Calcular estadísticas del grupo
        longitudes = [msg['num_bits'] for msg in msgs]
        tiempos = [msg['tiempo_s'] for msg in msgs]
        
        # Calcular intervalos entre transmisiones
        intervalos = []
        for i in range(1, len(tiempos)):
            intervalos.append(tiempos[i] - tiempos[i-1])
        
        # Verificar consistencia de longitud
        longitud_media = np.mean(longitudes)
        longitud_std = np.std(longitudes)
        
        es_consistente = longitud_std <= max_variacion_longitud
        
        sensor_info = {
            'id': sensor_id,
            'mensajes': msgs,
            'num_transmisiones': len(msgs),
            'longitud_media': longitud_media,
            'longitud_std': longitud_std,
            'es_consistente': es_consistente,
            'intervalo_medio': np.mean(intervalos) if intervalos else 0,
            'intervalo_std': np.std(intervalos) if intervalos else 0,
            'primer_tiempo_s': tiempos[0],
            'ultimo_tiempo_s': tiempos[-1]
        }
        
        sensores_tpms.append(sensor_info)
    
    # Ordenar por número de transmisiones (más probable = sensores reales)
    sensores_tpms.sort(key=lambda x: x['num_transmisiones'], reverse=True)
    
    return sensores_tpms, grupos_descartados


def generar_informe_filtrado(sensores_tpms, grupos_descartados, output_file='informe_filtrado.txt'):
    """
    Genera informe con solo los sensores TPMS válidos
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("INFORME DE SENSORES TPMS FILTRADOS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Sensores TPMS detectados: {len(sensores_tpms)}\n")
        f.write(f"Grupos descartados (ruido): {len(grupos_descartados)}\n")
        f.write("\n")
        
        # Resumen de sensores
        f.write("=" * 80 + "\n")
        f.write("RESUMEN DE SENSORES\n")
        f.write("=" * 80 + "\n\n")
        
        for i, sensor in enumerate(sensores_tpms, 1):
            f.write(f"SENSOR #{i}: {sensor['id']}\n")
            f.write(f"  Transmisiones:     {sensor['num_transmisiones']}\n")
            f.write(f"  Longitud media:    {sensor['longitud_media']:.1f} bits (±{sensor['longitud_std']:.1f})\n")
            f.write(f"  Intervalo medio:   {sensor['intervalo_medio']:.2f}s (±{sensor['intervalo_std']:.2f})\n")
            f.write(f"  Primera detección: {sensor['primer_tiempo_s']:.2f}s\n")
            f.write(f"  Última detección:  {sensor['ultimo_tiempo_s']:.2f}s\n")
            f.write(f"  Consistente:       {'✓ Sí' if sensor['es_consistente'] else '✗ No'}\n")
            f.write("\n")
        
        # Detalles de cada sensor
        for i, sensor in enumerate(sensores_tpms, 1):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"SENSOR #{i}: {sensor['id']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total de transmisiones: {sensor['num_transmisiones']}\n\n")
            
            # Listar todas las transmisiones
            for j, msg in enumerate(sensor['mensajes'], 1):
                f.write(f"--- Transmisión #{j} (Mensaje original #{msg['num']}) ---\n")
                f.write(f"Tiempo:  {msg['tiempo_ms']:.2f} ms ({msg['tiempo_s']:.2f} s)\n")
                f.write(f"Pulsos:  {msg['pulsos']}\n")
                f.write(f"Bits:    {msg['num_bits']}\n")
                f.write(f"Método:  {msg['metodo']}\n")
                f.write(f"Hex:     {msg['hex']}\n")
                
                # Mostrar binario en bloques de 64
                bits = msg['bits']
                f.write(f"Binario:\n")
                for k in range(0, len(bits), 64):
                    f.write(f"  {bits[k:k+64]}\n")
                
                f.write("\n")
        
        # Sección de ruido descartado
        if grupos_descartados:
            f.write("\n" + "=" * 80 + "\n")
            f.write("RUIDO/INTERFERENCIAS DESCARTADAS\n")
            f.write("=" * 80 + "\n\n")
            
            for sensor_id, msgs in sorted(grupos_descartados.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"ID: {sensor_id} - {len(msgs)} mensaje(s)\n")
                for msg in msgs[:3]:  # Mostrar solo primeros 3
                    f.write(f"  Tiempo: {msg['tiempo_s']:.2f}s, Hex: {msg['hex'][:40]}...\n")
                if len(msgs) > 3:
                    f.write(f"  ... y {len(msgs)-3} más\n")
                f.write("\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filtra y agrupa tramas TPMS del informe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  # Filtrar informe TPMS
  python3 filtrar_tpms.py informe_tpms.txt
  
  # Con parámetros personalizados
  python3 filtrar_tpms.py informe_tpms.txt --min-mensajes 5 --output sensores_filtrados.txt
  
  # Modo estricto (solo sensores muy consistentes)
  python3 filtrar_tpms.py informe_tpms.txt --min-mensajes 10 --max-variacion 5
        '''
    )
    
    parser.add_argument('informe', help='Archivo de informe TPMS a filtrar')
    parser.add_argument('-o', '--output', type=str, default='informe_filtrado.txt',
                        help='Archivo de salida (default: informe_filtrado.txt)')
    parser.add_argument('--min-mensajes', '-m', type=int, default=3,
                        help='Mínimo de mensajes para considerar sensor válido (default: 3)')
    parser.add_argument('--max-variacion', '-v', type=int, default=10,
                        help='Máxima variación de longitud permitida (default: 10)')
    
    args = parser.parse_args()
    
    # Parsear informe
    print(f"\n{'='*70}")
    print(f"📖 PARSEANDO INFORME TPMS")
    print(f"{'='*70}\n")
    print(f"Archivo: {args.informe}\n")
    
    mensajes = parsear_informe(args.informe)
    print(f"✅ {len(mensajes)} mensajes parseados\n")
    
    # Agrupar y filtrar
    sensores_tpms, grupos_descartados = agrupar_mensajes(
        mensajes,
        min_mensajes=args.min_mensajes,
        max_variacion_longitud=args.max_variacion
    )
    
    # Mostrar resumen en consola
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS DEL FILTRADO")
    print(f"{'='*70}\n")
    
    if len(sensores_tpms) == 0:
        print("❌ No se detectaron sensores TPMS válidos")
        print(f"\n💡 Sugerencias:")
        print(f"   - Reduce --min-mensajes (actual: {args.min_mensajes})")
        print(f"   - Aumenta --max-variacion (actual: {args.max_variacion})")
        return
    
    print(f"✅ {len(sensores_tpms)} sensor(es) TPMS detectado(s):\n")
    
    for i, sensor in enumerate(sensores_tpms, 1):
        print(f"🔹 Sensor #{i}: {sensor['id']}")
        print(f"   Transmisiones:   {sensor['num_transmisiones']}")
        print(f"   Longitud media:  {sensor['longitud_media']:.1f} bits")
        print(f"   Intervalo medio: {sensor['intervalo_medio']:.2f}s")
        print(f"   Período:         {sensor['primer_tiempo_s']:.1f}s - {sensor['ultimo_tiempo_s']:.1f}s")
        print()
    
    print(f"❌ {len(grupos_descartados)} grupo(s) descartado(s) como ruido\n")
    
    # Generar informe filtrado
    generar_informe_filtrado(sensores_tpms, grupos_descartados, args.output)
    
    print(f"{'='*70}")
    print(f"✅ INFORME FILTRADO GENERADO")
    print(f"{'='*70}")
    print(f"📄 Archivo: {args.output}")
    print(f"   Sensores TPMS: {len(sensores_tpms)}")
    print(f"   Ruido descartado: {len(grupos_descartados)} grupos")
    print()
    
    # Estadísticas finales
    total_mensajes_validos = sum(s['num_transmisiones'] for s in sensores_tpms)
    total_mensajes_descartados = sum(len(msgs) for msgs in grupos_descartados.values())
    
    print(f"📊 Estadísticas:")
    print(f"   Mensajes totales:     {len(mensajes)}")
    print(f"   Mensajes válidos:     {total_mensajes_validos} ({total_mensajes_validos/len(mensajes)*100:.1f}%)")
    print(f"   Mensajes descartados: {total_mensajes_descartados} ({total_mensajes_descartados/len(mensajes)*100:.1f}%)")
    print()


if __name__ == '__main__':
    main()
