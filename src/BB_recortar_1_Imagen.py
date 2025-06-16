OPENSLIDE_PATH = r'/usr/local/lib'

import os
import numpy as np
import cv2
from shutil import rmtree
from tqdm import tqdm
from openslide import open_slide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ExifTags
import warnings

def normalize_wsi(image):
    image_float = image.astype(np.float32)
    min_val = np.min(image_float)
    max_val = np.max(image_float)
    normalized_image = (image_float - min_val) / (max_val - min_val) * 255.0
    return np.clip(normalized_image, 0, 255).astype(np.uint8)

def Recortar_img(ruta_imagen, ruta_destino, tam_recorte=256, porcentaje_blanco=0.6, generar_rotaciones=False, mostrar_progreso=False, angle=0):
    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
    
    if not os.path.exists(ruta_imagen):
        print(f"Error: La imagen {ruta_imagen} no existe.")
        return -1
    
    try:
        imagen = open_slide(ruta_imagen)
        print(f"Imagen {nombre_imagen} abierta con OpenSlide")
    except OpenSlideError:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                imagen = Image.open(ruta_imagen)
                print(f"Imagen {nombre_imagen} abierta con PIL (posible formato no soportado por OpenSlide)")
        except Exception as e:
            print(f"Error al abrir la imagen {ruta_imagen}: {e}")
            return -1
    
    try:
        recortes = DeepZoomGenerator(imagen, tile_size=tam_recorte, overlap=0, limit_bounds=False)
        columnas, filas = recortes.level_tiles[recortes.level_count - 1]
    except Exception as e:
        print(f"Error al generar recortes para {nombre_imagen}: {e}")
        return -1
    
    total_recortes = columnas * filas
    
    with tqdm(total=total_recortes, desc="Procesando recortes", disable=not mostrar_progreso) as pbar:
        for fila in range(filas):
            for columna in range(columnas):
                try:
                    temp_tile = recortes.get_tile(recortes.level_count - 1, (columna, fila))
                    temp_tile = np.array(temp_tile)
                    
                    if temp_tile.shape[0] == tam_recorte and temp_tile.shape[1] == tam_recorte:
                        num_pixeles_blancos = np.sum(np.all(temp_tile >= [200, 200, 200], axis=-1))
                        if num_pixeles_blancos < (temp_tile.size * porcentaje_blanco / 3):
                            ruta_base = os.path.join(ruta_destino, f'{nombre_imagen}&{columna}_{fila}')
                            rotated_tile = temp_tile
                            if angle == 90:
                                rotated_tile = cv2.rotate(temp_tile, cv2.ROTATE_90_CLOCKWISE)
                            elif angle == 180:
                                rotated_tile = cv2.rotate(temp_tile, cv2.ROTATE_180)
                            elif angle == 270:
                                rotated_tile = cv2.rotate(temp_tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            cv2.imwrite(ruta_base + f'_{angle}.png', normalize_wsi(rotated_tile))
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Error en recorte ({columna}, {fila}) de {nombre_imagen}: {e}")
                    continue
    return 1