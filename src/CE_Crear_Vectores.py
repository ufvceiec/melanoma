import os
import configparser
from tqdm import tqdm
from pathlib import Path
import random
import shutil
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import torch

from src.CD_extraer_caracteristicas import Rutas_Archivos, Extraer_Caracteristicas
from src.BB_recortar_1_Imagen import Recortar_img
from src.CB_Train_Autoencoder import choose_best_model, Elegir_Dispositivo, CustomImageDataset
from src.CA_Model_Autoencoder import Autoencoder

from IPython.display import clear_output

torch.backends.cudnn.benchmark = True  # 🔥 Acelera operaciones en GPU

def Pacientes(ruta_carpeta):
    # Creamos una lista vacia
    archivos = []
    # Recorremos todos los archivos de la ruta de la carpeta indicada
    for archivo in tqdm(Path(ruta_carpeta).glob("**/*"), desc='cargando imagenes'):
        # Si la ruta es un archivo
        if archivo.is_file():
            # Añadimos a la lista el archivo
            archivos.append(archivo.name)
    # Devolvemos la lista
    return archivos

def Rutas_Archivos(ruta_carpeta):
    return [archivo for archivo in Path(ruta_carpeta).iterdir() if archivo.is_file()]


def Pacientes(ruta_vectores):
    return [archivo for archivo in Path(ruta_vectores).iterdir() if archivo.suffix == ".csv"]

def Ejecucion_Vectores(ruta_carpeta, skip = 0):
    ruta_vectores = "./vectores"
    if not os.path.exists(ruta_vectores):
        os.mkdir(ruta_vectores)
    ruta_recortes_aux = f"{ruta_vectores}/recortes_aux"
    ruta_vectores_aux = f"{ruta_vectores}/vectores_aux"
    
    imagenes_svs = Rutas_Archivos(ruta_carpeta)
    
    lista_vectores = Pacientes(ruta_vectores)
    
    # Crear un diccionario con los nombres base y la cantidad de rotaciones generadas
    nombres_vectores = {}
    for vector in lista_vectores:
        base_name = Path(vector).stem.rsplit("_", 1)[0]
        if base_name not in nombres_vectores:
            nombres_vectores[base_name] = set()
        nombres_vectores[base_name].add(Path(vector).stem.split("_")[-1])
    
    # Filtrar imagenes_svs y procesar solo aquellas que no tienen las 4 rotaciones completas
    imagenes_filtradas = []
    for imagen in imagenes_svs:
        base_name = imagen.stem
        if base_name not in nombres_vectores or len(nombres_vectores[base_name]) < 4:
            imagenes_filtradas.append(imagen)
    
    img_totales = len(imagenes_svs)
    print(f"Imagenes antes = {img_totales}")
    img_ya_filtradas = img_totales - len(imagenes_filtradas)
    print(f"Imagenes despues = {img_ya_filtradas}")
    
    if len(imagenes_filtradas) > skip:
        imagenes_filtradas = imagenes_filtradas[skip:]
        for imagen_svs in imagenes_filtradas:
            angles = [0, 90, 180, 270]
            base_name = imagen_svs.stem
            missing_angles = [angle for angle in angles if str(angle) not in nombres_vectores.get(base_name, set())]
            
            if not missing_angles:
                continue  # Si todas las rotaciones están presentes, omitir la imagen
            
            for angle in missing_angles:
                print(f"Imagenes procesadas = {img_ya_filtradas + skip}/{img_totales} | {round(((img_ya_filtradas + skip) / img_totales) * 100, 2)}%")
                print(f"Procesando : {imagen_svs} con rotación {angle}")
                if not os.path.exists(ruta_recortes_aux):
                    os.mkdir(ruta_recortes_aux)
                else:
                    shutil.rmtree(ruta_recortes_aux)
                    os.mkdir(ruta_recortes_aux)
                
                if not os.path.exists(ruta_vectores_aux):
                    os.mkdir(ruta_vectores_aux)
                else:
                    shutil.rmtree(ruta_vectores_aux)
                    os.mkdir(ruta_vectores_aux)
                print("Directorios limpios")
                
                aux = Recortar_img(imagen_svs, ruta_recortes_aux, tam_recorte=256, porcentaje_blanco=0.6, generar_rotaciones=True, mostrar_progreso=True, angle=angle)
                if aux != -1:
                    print(f'{ruta_vectores}/{str(imagen_svs.name)[:-4]}_{angle}.csv')
                    Extraer_Caracteristicas(ruta_recortes_aux=ruta_recortes_aux, 
                                            ruta_vectores_aux=ruta_vectores_aux, 
                                            csv_output_path=f'{ruta_vectores}/{str(imagen_svs.name)[:-4]}_{angle}.csv', 
                                            batch_size=32)
                else:
                    skip += 1
                clear_output(wait=True)
            img_ya_filtradas += 1

def Crear_Vectores(skip = 0):
    config = configparser.ConfigParser()
    
    config.read('./config.cfg')
    # Extraemos la carpeta de imagenes sin procesar
    rutas_carpetas = []
    rutas_carpetas.append(config['Imagenes_NP']['carpeta_imagenes_correctas'])
    rutas_carpetas.append(config['Imagenes_NP']['carpeta_imagenes_multiples_cortes'])
    rutas_carpetas.append(config['Imagenes_NP']['carpeta_imagenes_multiples_cortes_correctos'])
    
    for ruta_carpeta in rutas_carpetas:
        Ejecucion_Vectores(ruta_carpeta,skip)
