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

from src.BB_recortar_1_Imagen import Recortar_img

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

def Crear_Recortes(ruta_origen, ruta_destino, tiny_set=-1):
    # Creamos un objeto configparser
    config = configparser.ConfigParser()
    # Leemos el fichero de configuracion
    if os.name == "nt":
        config.read('./config_Windows.cfg')
    elif os.name == "posix":
        config.read('./config.cfg')
    else:
        print("ERROR: No existe archivo de configuracion para este sistema operativo")

    # Cargamos el tamaño del recorte
    tamano = config.getint('Parametros', 'tam_recorte')
    #Cargamos el porcentaje de blanco 
    porcentaje_blanco = config.getfloat('Parametros', 'porcentaje_blanco')

    # Verificar si la ruta de origen existe
    if not os.path.exists(ruta_origen):
        # Si no existe no podemos continuar, imprime un mensaje de error y termina la ejecucion 
        print(f"La ruta {ruta_origen} no existe.")
        return
        
    Recortar_img(ruta_origen, ruta_destino, tamano, porcentaje_blanco)

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