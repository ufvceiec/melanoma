import os
import configparser
from tqdm import tqdm
from pathlib import Path
import random
import shutil
import pandas as pd
import gc

from src.BA_Crear_Recortes import Crear_Recortes

#-------------------------------------------------------------------------------------------------------------

def Extraer_Archivos(ruta_carpeta):
    # Creamos una lista vacia
    archivos = []
    # Recorremos todos los archivos de la ruta de la carpeta indicada
    for archivo in tqdm(Path(ruta_carpeta).glob("**/*"), desc='cargando imagenes'):
        # Si la ruta es un archivo
        if archivo.is_file():
            # Añadimos a la lista el archivo
            archivos.append(archivo)
    # Devolvemos la lista
    return archivos

#-------------------------------------------------------------------------------------------------------------

def Extraer_Nombres_Pacientes(ruta_carpeta):
    # Extraemos una lista de rutas de archivos de la carpeta seleccionada
    names = Extraer_Archivos(ruta_carpeta)
    # Recorremos la lista
    # Extraemos el nombre del archivo desde la ruta
    # Aplicamos normalizacion para sacar el ID de paciente
    names = [(archivo.name.split('_')[0]).split('.')[0] for archivo in names]
    # Devolvemos la lista de nombres de pacientes
    return names

#-------------------------------------------------------------------------------------------------------------

def Numero_ImagenesXPaciente(ruta_carpeta):
    # Creamos un diccionario vacio
    Img_X_Paciente = {}
    # Recorremos la lista de nombres de pacientes
    for nombre in Extraer_Nombres_Pacientes(ruta_carpeta):
        # Por cada nombre aumentamos en 1 el contador de imagenes que tiene ese paciente
        Img_X_Paciente[nombre] = Img_X_Paciente.get(nombre, 0) + 1
    # Devolvemos el diccionario
    return Img_X_Paciente

#-------------------------------------------------------------------------------------------------------------

def Distribuir_Pacientes(ruta_carpeta):
    # Extraemos la lista de rutas de archivos
    archivos = Extraer_Archivos(ruta_carpeta)
    # Extraemos los nombres de esos archivos
    names = Extraer_Nombres_Pacientes(ruta_carpeta)
    # Extraemos cuantas imagenes tiene cada paciente
    Img_X_Paciente = Numero_ImagenesXPaciente(ruta_carpeta)

    # Indicamos el tamaño que tiene que tener cada set
    len_test_I = int(len(archivos) * 0.2)
    len_val_I = int(len(archivos) * 0.2)
    len_train_I = len(archivos) - (len_test_I + len_val_I)

    # Creamos 3 listas vacias una por set de entrenamiento
    train, val, test = [], [], []
    # Creamos 3 centinelas a 0 para comprobar el numero de imagenes por set
    total_imgs_train = total_imgs_val = total_imgs_test = 0

    # Sacamos los pacientes unicos
    pacientes = list(Img_X_Paciente.keys())
    # Randomizamos el nombre de los pacientes
    random.shuffle(pacientes)

    # Recorremos el listado de pacientes
    for paciente in pacientes:
        # Extraemos el numero de imagenes por paciente
        imgs = Img_X_Paciente[paciente]

        # Indicamos donde van cada paciente
        if total_imgs_train + imgs <= len_train_I:
            train.append(paciente)
            total_imgs_train += imgs
        elif total_imgs_val + imgs <= len_val_I:
            val.append(paciente)
            total_imgs_val += imgs
        elif total_imgs_test + imgs <= len_test_I:
            test.append(paciente)
            total_imgs_test += imgs
            
        if (total_imgs_train >= len_train_I and 
            total_imgs_val >= len_val_I and 
            total_imgs_test >= len_test_I):
            break
            
    return train, val, test

#-------------------------------------------------------------------------------------------------------------

def Cargar_Rutas_Datasets():
    import os
    # Si el fichero de configuracion no existe no continua el programa
    
    # Creamos un objeto configparser
    config = configparser.ConfigParser()
    # Leemos el fichero de configuracion
    import os

    
    config.read('./config.cfg')
   

    # Extraemos la carpeta de imagenes sin procesar
    ruta_carpetas = []
    ruta_carpetas.append(config['Imagenes_NP']['carpeta_imagenes_correctas'])
    ruta_carpetas.append(config['Imagenes_NP']['carpeta_imagenes_multiples_cortes'])
    ruta_carpetas.append(config['Imagenes_NP']['carpeta_imagenes_multiples_cortes_correctos'])

    # Creamos 3 listas vacias para almacenar las rutas de las imagenes
    rutas_train, rutas_val, rutas_test = [], [], []

    for ruta_carpeta in ruta_carpetas:
        # Sacamos un listado de los pacientes que van a cada set
        train, val, test = Distribuir_Pacientes(ruta_carpeta)
        # Extraemos las rutas de los archivos de la carpeta
        archivos = Extraer_Archivos(ruta_carpeta)
    
        # Recorremos los archivos con los nombres de los pacientes
        for archivo in archivos:
            # Extraemos el nombre del paciente de dicho archivo
            name = (archivo.name.split('_')[0]).split('.')[0]
            # Si el paciente se encuentra en alguno de los sets se alamacena la ruta en dicho set
            if name in train:
                rutas_train.append(archivo)
            elif name in val:
                rutas_val.append(archivo)
            elif name in test:
                rutas_test.append(archivo)
        
        del archivos
        gc.collect
    
    return rutas_train, rutas_val, rutas_test

#-------------------------------------------------------------------------------------------------------------

def Crear_Datasets():
    # Si el archivo de configuracion no existe para el proceso
    if not os.path.exists('./config.cfg'):
        raise FileNotFoundError("El archivo de configuración 'config.cfg' no se encontró.")
    # Creamos un arvhivo configparser
    config = configparser.ConfigParser()
    # Leemos el archivo de configuracion
    config.read('./config.cfg')

    # Cargamos las rutas de destino para cada set
    ruta_train = Path(config['Carpetas']['carpeta_train'])
    ruta_val = Path(config['Carpetas']['carpeta_val'])
    ruta_test = Path(config['Carpetas']['carpeta_test'])

    ruta_destino_train = Path(config['Carpetas']['carpeta_recortes_train'])
    ruta_destino_val = Path(config['Carpetas']['carpeta_recortes_val'])
    ruta_destino_test = Path(config['Carpetas']['carpeta_recortes_test'])
    
    # Eliminamos y volvemos a crear las carpetas vacías
    for ruta in tqdm([ruta_train, ruta_val, ruta_test, ruta_destino_train, ruta_destino_val, ruta_destino_test], desc='Limpiando directorios'):
        if ruta.exists():
            shutil.rmtree(ruta)
        ruta.mkdir(parents=True, exist_ok=True)
    # Cargamos las rutas de imagenes de cada set
    """
    rutas_train, rutas_val, rutas_test = Cargar_Rutas_Datasets()
   
    # Recorremos las rutas de las imagenes de cada set copiandolas en la carpeta indicada
    for ruta in tqdm(rutas_train, desc='Copiando train'):
        if ruta.is_file():
            Crear_Recortes(ruta, ruta_destino_train, tiny_set=-1)
    
    for ruta in tqdm(rutas_val, desc='Copiando val'):
        if ruta.is_file():
            Crear_Recortes(ruta, ruta_destino_val, tiny_set=-1)

    for ruta in tqdm(rutas_test, desc='Copiando test'):
        if ruta.is_file():
            Crear_Recortes(ruta, ruta_destino_test, tiny_set=-1)
    """
#-------------------------------------------------------------------------------------------------------------
