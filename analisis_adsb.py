#!/usr/bin/env python3
import pyModeS as pms
import pandas as pd

# Diccionario para agrupar la información unificada por cada aeronave (dirección ICAO)
aeronaves = {}

print("Procesando mensajes ADS-B desde '../frames.txt'...")

try:
    for line in open('../frames.txt'):
        msg = line.strip().strip('*;')
        
        # Saltamos líneas que no cumplan con la longitud estándar de un mensaje largo (28 caracteres hex)
        if len(msg) != 28:
            continue
            
        try:
            # Usamos la API unificada de pyModeS que decodifica y calcula el CRC automáticamente
            decoded = pms.decode(msg)
            
            # Validamos que sea un formato ADS-B (DF17) y que pase el control de errores CRC
            if decoded.get("df") != 17 or not decoded.get("crc_valid", True):
                continue
                
            icao = decoded.get("icao")
            if not icao:
                continue
                
            # Inicializamos la aeronave en el registro unificado si no existía antes
            if icao not in aeronaves:
                aeronaves[icao] = {
                    'ICAO': icao, 
                    'Indicativo': 'Desconocido', 
                    'Altitud (ft)': 'N/A', 
                    'Velocidad (kt)': 'N/A'
                }
            
            # Extraemos los campos calculados automáticamente según el tipo de mensaje
            if "callsign" in decoded and decoded["callsign"]:
                aeronaves[icao]['Indicativo'] = decoded["callsign"].strip()
                
            if "altitude" in decoded and decoded["altitude"] is not None:
                aeronaves[icao]['Altitud (ft)'] = decoded["altitude"]
                
            if "speed" in decoded and decoded["speed"] is not None:
                aeronaves[icao]['Velocidad (kt)'] = decoded["speed"]
                
        except Exception:
            # Si un mensaje corrupto causa un fallo interno en la librería, pasamos al siguiente
            continue

except FileNotFoundError:
    print("Error: No se encontró el archivo '../frames.txt'. Verifica la ubicación del archivo.")
    exit(1)

# --- Renderizado de la Tabla Final ---
lista_aeronaves = list(aeronaves.values())

if lista_aeronaves:
    df_aviones = pd.DataFrame(lista_aeronaves)
    df_aviones = df_aviones[['ICAO', 'Indicativo', 'Altitud (ft)', 'Velocidad (kt)']]
    
    print("\n=== TABLA DE AERONAVES DETECTADAS ===")
    print(df_aviones.to_string(index=False))
    
    # Exportación limpia para tu informe
    df_aviones.to_csv('tabla_aviones_adsb.csv', index=False)
    print("\n[✓] Tabla exportada correctamente a 'tabla_aviones_adsb.csv'")
else:
    print("\n[!] No se pudieron agrupar suficientes tramas completas para armar la tabla.")
