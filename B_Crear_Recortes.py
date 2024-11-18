OPENSLIDE_PATH = r'/usr/local/lib'

import os
import shutil
import numpy as np
import cv2
#import staintools
from shutil import rmtree
import configparser
from tqdm import tqdm
import random

os.environ['PATH'] = OPENSLIDE_PATH + ";" + os.environ['PATH']

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide import open_slide
        from openslide.deepzoom import DeepZoomGenerator
        from openslide import OpenSlideError
else:
    import openslide
    from openslide import open_slide
    from openslide.deepzoom import DeepZoomGenerator
    from openslide import OpenSlideError

#------------------------------------------------------------------------------------------------------------

def normalize_wsi(image):
    # Pasamos la imagen a float
    image_float = image.astype(np.float32)
    # Aplanamos la imagen para tener un vector
    flat_image = image_float.flatten()
    # Extraemos el valor minimo
    min_val = np.min(flat_image)
    # Extraemos el valor maximo
    max_val = np.max(flat_image)

    # Aplicar la normalización
    normalized_image = (image_float - min_val) / (max_val - min_val) * 255.0

    # Convertir de nuevo a uint8
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)

    # Devolvemos la imagen
    return normalized_image
    
#------------------------------------------------------------------------------------------------------------

def Recortar_img(ruta_imagen, ruta_destino, tam_recorte=256, porcentaje_blanco=0.6):
    # Extraer el nombre de la imagen, quitamos la extension (.svs)
    nombre_imagen = os.path.split(ruta_imagen)[1][:-4]

    try:
        # Cargar la imagen (.svs)
        imagen = open_slide(ruta_imagen)
    except OpenSlideError as e:
        print(f"Error al abrir la imagen {ruta_imagen}: {e}")
        return
        
    try:
        # Crear los recortes de dicha imagen
        recortes = DeepZoomGenerator(imagen, tile_size=tam_recorte, overlap=0, limit_bounds=False)
        # Indicamos el numero de columnas y filas que tiente la imagen
        columnas, filas = recortes.level_tiles[recortes.level_count - 1]
    except Exception as e:
        print(f"Error al generar recortes para {nombre_imagen}: {e}")
        return
        
    # Recorremos cada recorte realizado
    for fila in tqdm(range(filas), desc=f'Imagen: {nombre_imagen}'):
        for columna in range(columnas):
            try:
                # Cargamos el recorte
                temp_tile = recortes.get_tile(recortes.level_count - 1, (columna, fila))
                temp_tile = np.array(temp_tile)
            except OpenSlideError as e:
                print(f"Error en recorte ({columna}, {fila}) de {nombre_imagen}: {e}")
                return

            # Verificamos que el recorte tiene las dimensiones adecuadas
            if temp_tile.shape[0] == tam_recorte and temp_tile.shape[1] == tam_recorte:
                # Extraemos el numero de pixeles en blanco ( < 200 )
                num_pixeles_blancos = np.sum(np.all(temp_tile >= [200, 200, 200], axis=-1))
                # Si el numero de pixeles en blanco es menor al porcentaje indicado
                if num_pixeles_blancos < (temp_tile.size * porcentaje_blanco / 3):
                    try:
                        # Cargamos la ruta de la imagen destino 
                        ruta_recorte = os.path.join(ruta_destino, f'{nombre_imagen}&{columna}_{fila}.png')
                        # Guarfamos el recorte en la ruta indicada
                        cv2.imwrite(ruta_recorte, normalize_wsi(temp_tile))
                    except Exception as e:
                        print(f"Error al guardar el recorte ({columna}, {fila}) de {nombre_imagen}: {e}")
                        return

#------------------------------------------------------------------------------------------------------------

def Crear_Recortes(ruta_origen, ruta_destino, tiny_set=-1):
    # Cargamos la ruta de destino para los recortes (ruta origen + 'recortes')
    ruta_destino = os.path.join(ruta_origen, 'recortes')
    # Creamos un objeto configparser
    config = configparser.ConfigParser()
    # Leemos el fichero de configuracion
    config.read('./config.cfg')

    # Cargamos el tamaño del recorte
    tamano = config.getint('Parametros', 'tam_recorte')
    #Cargamos el porcentaje de blanco 
    porcentaje_blanco = config.getfloat('Parametros', 'porcentaje_blanco')

    # Verificar si la ruta de origen existe
    if not os.path.exists(ruta_origen):
        # Si no existe no podemos continuar, imprime un mensaje de error y termina la ejecucion 
        print(f"La ruta {ruta_origen} no existe.")
        return
    
    # Obtener archivos .svs en la ruta de origen
    img_dir = [archivo for archivo in os.listdir(ruta_origen) if archivo.endswith('.svs')]
    
    # Si tiny_set está especificado, extraer un número reducido de imágenes
    if tiny_set > 0:
        img_dir = random.sample(img_dir, min(tiny_set, len(img_dir)))  # Extrae hasta tiny_set imágenes
    
    # Crear la carpeta de destino para los recortes, eliminando la existente si es necesario
    if os.path.exists(ruta_destino):
        shutil.rmtree(ruta_destino)  # Eliminar la carpeta existente
        
    os.makedirs(ruta_destino)  # Crear la nueva carpeta

    # Procesar cada imagen
    for imagen in img_dir:
        # Creamos los recortes de imagenes 
        Recortar_img(os.path.join(ruta_origen, imagen), ruta_destino, tamano, porcentaje_blanco)
        # Eliminamos la imagen original (no sobrecargamos memoria)
        os.remove(os.path.join(ruta_origen, imagen))

#------------------------------------------------------------------------------------------------------------

def Ejecutar_Crear_Recortes(tiny_set = True):
    # Creamos un objeto configparser
    config = configparser.ConfigParser()
    # Leemos el fichero de configuracion
    config.read('./config.cfg')
    # Indicamos si creamos un set de prueba o no
    if tiny_set:
        Crear_Recortes(config['Carpetas']['carpeta_train'], config['Carpetas']['carpeta_recortes_train'], tiny_set=6)
        Crear_Recortes(config['Carpetas']['carpeta_val'], config['Carpetas']['carpeta_recortes_val'], tiny_set=2)
        Crear_Recortes(config['Carpetas']['carpeta_test'],config['Carpetas']['carpeta_recortes_test'],  tiny_set=2)
    else:
        Crear_Recortes(config['Carpetas']['carpeta_train'], config['Carpetas']['carpeta_recortes_train'], tiny_set=-1)
        Crear_Recortes(config['Carpetas']['carpeta_val'], config['Carpetas']['carpeta_recortes_val'], tiny_set=-1)
        Crear_Recortes(config['Carpetas']['carpeta_test'],config['Carpetas']['carpeta_recortes_test'],  tiny_set=-1)
