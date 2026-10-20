#!/usr/bin/env python3
import numpy as np

def crc24(bits):
    # Función CRC provista por el enunciado
    reg = 0
    for b in bits:
        reg = (reg << 1) | b
        if reg & (1 << 24): 
            reg ^= 0x1FFF409
    return reg  # 0 = trama válida

def bits_a_hex(bits):
    # Convierte una lista de 112 bits (0/1) a una cadena hexadecimal de 28 caracteres
    trama_hex = ""
    for i in range(0, len(bits), 4):
        cuarteto = bits[i:i+4]
        valor = 0
        for bit in cuarteto:
            valor = (valor << 1) | bit
        trama_hex += f"{valor:X}"
    return trama_hex

if __name__ == '__main__':
    # 1. Leer el fichero binario (Ajusta la ruta si es necesario)
    ruta_bin = '/home/diego/Descargas/dump1090/testfiles/modes1.bin'
    try:
        raw = np.fromfile(ruta_bin, np.uint8)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {ruta_bin}")
        exit(1)
        
    # Convertir muestras IQ a magnitud analógica
    iq = (raw[0::2] - 127.5) + 1j * (raw[1::2] - 127.5)
    mag = np.abs(iq)
    
    tramas_propias = set()
    
    # 2. Ventana de escaneo a lo largo de todo el vector de magnitud
    # Necesitamos espacio para 16 muestras de preámbulo + 224 muestras de datos (112 bits * 2)
    longitud_trama_muestras = 16 + (112 * 2)
    
    print("Escaneando el archivo en busca de tramas Mode S...")
    
    for i in range(len(mag) - longitud_trama_muestras):
        # Umbral dinámico simple para evitar procesar ruido de fondo
        if mag[i] < 10: 
            continue
            
        # Comprobar la estructura básica del preámbulo en las posiciones relativas: 0, 2, 7 y 9
        # Buscamos que los picos del preámbulo sean superiores a sus muestras adyacentes de espacio
        p0, p2, p7, p9 = mag[i], mag[i+2], mag[i+7], mag[i+9]
        p1, p3, p8, p10 = mag[i+1], mag[i+3], mag[i+8], mag[i+10]
        
        if p0 > p1 and p2 > p3 and p7 > p8 and p9 > p10:
            # 3. Si parece un preámbulo válido, decodificamos los 112 bits
            bits = []
            inicio_datos = i + 16
            
            for k in range(112):
                muestras_bit = inicio_datos + (2 * k)
                if mag[muestras_bit] > mag[muestras_bit + 1]:
                    bits.append(1)
                else:
                    bits.append(0)
            
            # 4. Validar la integridad de la trama con el CRC24
            if crc24(bits) == 0:
                trama_hex = bits_a_hex(bits)
                tramas_propias.add(trama_hex)
                # Saltamos toda la trama para no volver a evaluar las mismas muestras
                i += longitud_trama_muestras 

    # 5. Cargar tramas de dump1090 para realizar la comparación
    tramas_dump1090 = set()
    ruta_frames = '../frames.txt'
    try:
        for line in open(ruta_frames):
            trama = line.strip().strip('*;')
            if len(trama) == 28: # Filtrar solo tramas largas de 112 bits
                tramas_dump1090.add(trama.upper())
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_frames}. Ejecuta el Paso 1 primero.")
        exit(1)

    # 6. Comparación estadística de resultados
    coincidentes = tramas_propias.intersection(tramas_dump1090)
    solo_propias = tramas_propias - tramas_dump1090
    solo_dump1090 = tramas_dump1090 - tramas_propias

    print("\n================ REPORTES DE COMPARACIÓN ================")
    print(f"Tramas totales encontradas por dump1090 (112 bits): {len(tramas_dump1090)}")
    print(f"Tramas totales encontradas por tu demodulador:      {len(tramas_propias)}")
    print(f"Tramas idénticas que coinciden en ambos:            {len(coincidentes)}")
    print(f"Tramas que TU algoritmo descubrió y dump1090 NO:    {len(solo_propias)}")
    print(f"Tramas que dump1090 descubrió y TU algoritmo NO:    {len(solo_dump1090)}")
    print("=========================================================")
