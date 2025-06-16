import os
import json
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import re
from collections import defaultdict
import itertools
import pandas as pd
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import shutil
import gc

from src.CA_Model_Autoencoder import Autoencoder
from src.CA_Model_Autoencoder import train_autoencoder

import random
import matplotlib.pyplot as plt

import ast

#--------------------------------------------------------------------------------------------------------------------------------------------------------

def display_best_average_worst_models(results_path="results.json", device=None, data_loader=None, num_images=3):
    """
    Selecciona el mejor modelo (menor pérdida), el modelo en la media, y el peor modelo (mayor pérdida),
    y muestra imágenes originales y reconstruidas. Calcula y muestra la varianza entre las pérdidas.

    Args:
        results_path (str): Ruta al archivo JSON con los resultados de los modelos.
        device (torch.device): Dispositivo para cargar el modelo (CPU o GPU).
        data_loader (DataLoader): Cargador de datos para generar imágenes.
        num_images (int): Número de imágenes a mostrar por modelo.
    """
    if not os.path.exists(results_path):
        print(f"Error: El archivo de resultados {results_path} no existe.")
        return
    
    # Cargar los resultados guardados en el JSON
    with open(results_path, "r") as f:
        results = json.load(f)

    # Crear una lista de pérdidas de test con los nombres de los modelos
    model_losses = [(model_name, metrics["test_loss_MSE"]) for model_name, metrics in results.items()]
    
    # Ordenar los modelos por pérdida de test
    model_losses.sort(key=lambda x: x[1])  # Orden ascendente por test_loss
    
    # Seleccionar el mejor, el peor y el modelo más cercano a la media
    best_model = model_losses[0]  # El menor
    worst_model = model_losses[-1]  # El mayor
    mean_loss = np.mean([loss for _, loss in model_losses])  # Promedio de pérdidas
    
    # Encontrar el modelo más cercano al promedio
    average_model = min(model_losses, key=lambda x: abs(x[1] - mean_loss))

    # Diccionario de modelos seleccionados
    selected_models = {
        "Mejor": best_model,
        "Media": average_model,
        "Peor": worst_model,
    }
    
    # Generar imágenes representativas para cada modelo seleccionado
    for category, (model_name, test_loss) in selected_models.items():
        metrics = results[model_name]  # Obtener métricas del modelo seleccionado
        
        # Obtener pérdidas
        train_loss = metrics["train_losses"]
        val_loss = metrics["val_losses"]
        
        # Calcular la varianza entre las pérdidas
        losses = np.array([train_loss, val_loss, test_loss])
        variance = np.var(losses)
        
        print(f"Generando imágenes para el modelo '{model_name}' ({category})")
        print(f"  - Test Loss: {test_loss:.6f}")
        print(f"  - Varianza entre pérdidas: {variance:.6f}")
        
        # Cargar parámetros del modelo
        num_kernels = metrics["num_kernels"]
        depth = metrics["depth"]
        learning_rate = metrics["learning_rate"]

        param_str = f"nk{num_kernels}_d{depth}_lr{learning_rate}"
        weight_path = f"../pesos/autoencoder_{param_str}.pth"

        if not os.path.exists(weight_path):
            print(f"Error: no se encontraron los pesos en {weight_path}")
            continue
        
        # Cargar modelo
        autoencoder = Autoencoder(num_kernels=num_kernels, depth=depth).to(device)
        state_dict = torch.load(weight_path, map_location=device)
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        
        # Mostrar imágenes reconstruidas usando la función auxiliar
        mostrar_imagenes_reconstruidas(autoencoder, data_loader, device, num_imagenes=num_images)

#------------------------------------------------------------------------------------------------------------

def Elegir_Dispositivo(solo_CPU=True):
    if solo_CPU:
        print("Has elegido CPU")
        device = torch.device("cpu")
    elif not torch.cuda.is_available():  # Si solo queremos CPU o no hay GPU disponible
        print("No hay graficas disponibles")
        device = torch.device("cpu")
    else:
        print("Eligiendo mejor GPU disponible...")
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            device = torch.device("cpu")  # Evita fallos si no hay GPU
        else:
            max_memory_gpu = None
            max_free_memory = 0
            for i in range(num_gpus):
                try:
                    free_memory, total_memory = torch.cuda.mem_get_info(i)
                    print(f'GPU {i}: Memoria libre: {free_memory / 1024**3:.2f} GB de {total_memory / 1024**3:.2f} GB totales')
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        max_memory_gpu = i
                except RuntimeError:
                    print(f"No se pudo obtener información de GPU {i}")

            if max_memory_gpu is not None:
                device = torch.device(f'cuda:{max_memory_gpu}')
                torch.cuda.set_device(device)
            else:
                device = torch.device("cpu")  # Si ninguna GPU es válida, usar CPU
        
    print(f"Usando dispositivo: {device}")
    return device

#------------------------------------------------------------------------------------------------------------

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


#------------------------------------------------------------------------------------------------------------

def mostrar_imagenes_reconstruidas(autoencoder, data_loader, device, num_imagenes=3):
    # Configurar autoencoder en modo evaluación
    autoencoder.eval()
    
    # Obtener un batch del data loader
    data_iter = iter(data_loader)
    imagenes, _ = next(data_iter)
    imagenes = imagenes[:num_imagenes].to(device)  # Seleccionar las primeras imágenes
    
    # Generar las imágenes reconstruidas
    with torch.no_grad():
        reconstruidas = autoencoder(imagenes).cpu().numpy()
    
    # Pasar imágenes originales a numpy
    originales = imagenes.cpu().numpy()
    
    # Crear subplots para mostrar imágenes
    fig, axes = plt.subplots(num_imagenes, 2, figsize=(10, 5 * num_imagenes))
    
    # Si solo es una imagen, asegúrate de que axes sea un array bidimensional
    if num_imagenes == 1:
        axes = [axes]  # Convertir en una lista para iterar
    
    for i in range(num_imagenes):
        original = originales[i].transpose(1, 2, 0)  # Cambiar formato (C, H, W) → (H, W, C)
        reconstruida = reconstruidas[i].transpose(1, 2, 0)
        
        # Limitar valores entre 0 y 1
        original = np.clip(original, 0, 1)
        reconstruida = np.clip(reconstruida, 0, 1)
        
        # Mostrar la imagen original
        axes[i][0].imshow(original, cmap="gray" if original.ndim == 2 else None)
        axes[i][0].set_title("Original")
        axes[i][0].axis("off")
        
        # Mostrar la imagen reconstruida
        axes[i][1].imshow(reconstruida, cmap="gray" if reconstruida.ndim == 2 else None)
        axes[i][1].set_title("Reconstruida")
        axes[i][1].axis("off")
    
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------------------------------

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    train_dir = config.get('Carpetas', 'carpeta_recortes_train')
    val_dir = config.get('Carpetas', 'carpeta_recortes_val')
    test_dir = config.get('Carpetas', 'carpeta_recortes_test')
    
    num_kernels = eval(config.get('Parametros', 'num_kernels'))
    depths = eval(config.get('Parametros', 'depths'))
    learning_rates = eval(config.get('Parametros', 'learning_rates'))
    batch_size = config.getint('Parametros', 'batch_size')
    num_epochs = config.getint('Parametros', 'num_epochs')
    
    return train_dir, val_dir, test_dir, num_kernels, depths, learning_rates, batch_size, num_epochs

#------------------------------------------------------------------------------------------------------------

def choose_best_model(results_path="../results.json", device=None):
    """
    Selecciona el mejor modelo basado en las métricas MSE de test, varianza entre las pérdidas
    y clasifica los errores en bueno, intermedio y malo.
    
    Args:
        results_path (str): Ruta al archivo JSON con los resultados de los modelos.
        device (torch.device): Dispositivo para cargar el modelo (CPU o GPU).
    
    Returns:
        best_autoencoder (nn.Module): El mejor modelo cargado.
        best_params (dict): Los hiperparámetros del mejor modelo.
        best_test_loss (float): Pérdida de test del mejor modelo.
    """
    if not os.path.exists(results_path):
        print(f"Error: El archivo de resultados {results_path} no existe.")
        return None, None, None

    # Cargar los resultados guardados en el JSON
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Preparar una lista para evaluar cada modelo
    evaluations = []

    for model_name, metrics in results.items():
        test_loss = metrics["test_loss_MSE"]
        train_loss = metrics["train_losses"]
        val_loss = metrics["val_losses"]
        
        # Calcular la varianza entre las pérdidas
        losses = np.array([train_loss, val_loss, test_loss])
        variance = np.var(losses)

        # Clasificar el error basado en la pérdida de test
        if test_loss < 0.005:  # Umbral para "bueno"
            error_class = "Bueno"
        elif test_loss < 0.02:  # Umbral para "intermedio"
            error_class = "Intermedio"
        else:  # Clasificación "malo"
            error_class = "Malo"

        evaluations.append({
            "model_name": model_name,
            "test_loss": test_loss,
            "variance": variance,
            "error_class": error_class,
            "params": metrics
        })

    # Elegir el mejor modelo basado en el menor MSE de test
    best_model = min(evaluations, key=lambda x: x["test_loss"])

    # Mostrar las métricas del mejor modelo
    best_params = best_model["params"]
    best_test_loss = best_model["test_loss"]
    best_variance = best_model["variance"]
    best_error_class = best_model["error_class"]

    print(f"Mejor modelo encontrado: {best_model['model_name']}")
    print(f"Pérdida de Test: {best_test_loss}")
    print(f"Varianza entre Train, Val y Test: {best_variance}")
    print(f"Clasificación del Error: {best_error_class}")
    print(f"Parámetros: {best_params}")

    # Cargar los pesos del mejor modelo
    param_str = f"nk{best_params['num_kernels']}_d{best_params['depth']}_lr{best_params['learning_rate']}"
    weight_path = f"./pesos/autoencoder_{param_str}.pth"
    
    if not os.path.exists(weight_path):
        print(f"Error: no se encontraron los pesos en {weight_path}")
        return None, None, None

    
    
    # Crear el modelo y cargar los pesos
    best_autoencoder = Autoencoder(
        num_kernels=best_params['num_kernels'], 
        depth=best_params['depth']
    ).to(device)

    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model_dict = best_autoencoder.state_dict()
    
    # Filtrar solo las capas que coinciden en nombres y formas
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    # Actualizar el modelo con los pesos filtrados
    model_dict.update(filtered_dict)
    best_autoencoder.load_state_dict(model_dict, strict=False)  # strict=False permite omitir capas incompatibles

    best_autoencoder.eval()

    print(f"Pesos cargados desde: {weight_path}")

    return best_autoencoder, best_params, {
        "test_loss": best_test_loss,
        "variance": best_variance
    }




#------------------------------------------------------------------------------------------------------------

def grid_search(train_dir, val_dir, test_dir, param_grid, num_epochs=20, batch_size=32, filename="resultados.csv"):
    # Transformaciones para los datos
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    val_dataset = CustomImageDataset(val_dir, transform=transform)
    test_dataset = CustomImageDataset(test_dir, transform=transform)
    
    # Cargar datasets en dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Lista para almacenar resultados de todas las configuraciones
    results = []
    
    # Iterar sobre todas las combinaciones de hiperparámetros
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Probando configuración: {param_dict}")
       
        torch.cuda.empty_cache()
        
        # Seleccionar dispositivo
        device = Elegir_Dispositivo(solo_CPU=False)
        
        # Crear modelo Autoencoder
        autoencoder = Autoencoder(num_kernels=param_dict['num_kernels'], depth=param_dict['depth']).to(device)
        
        # Entrenar el modelo y obtener resultados
        results_epoch_train, results_epoch_test = train_autoencoder(
            device, autoencoder, train_loader, val_loader, test_loader, 
            num_epochs=num_epochs, learning_rate=param_dict['learning_rate'], param_dict=param_dict
        )
        
        # Consolidar resultados en un diccionario
        results_train = {
            "train_losses": results_epoch_train["train_losses"][-1],
            "val_losses": results_epoch_train["val_losses"][-1]
        }
        results_test = {
            "test_loss_MSE": results_epoch_test["test_loss_MSE"][-1].item(),  # Convertir tensor a escalar
            "test_loss_SSIM": results_epoch_test["test_loss_SSIM"][-1].item()
        }
        resultados = {**param_dict, **results_train, **results_test}
        resultados_autoencoder = {}
        if os.path.exists('resultados_autoencoder.json'):
            with open('resultados_autoencoder.json', 'r') as file:
                resultados_autoencoder = json.load(file)
        kernels = param_dict['num_kernels']
        depth = param_dict['depth']
        lr = param_dict['learning_rate']
        resultados_autoencoder[f'num_kern_{kernels}_depth_{depth}_lr_{lr}'] = resultados
        
        with open('resultados_autoencoder.json', 'w') as file:
            json.dump(resultados_autoencoder, file, indent=4)  # Usamos indent para mejorar la legibilidad
        
        # Liberar memoria de GPU
        del autoencoder
        
        torch.cuda.empty_cache()
    
    # Guardar resultados en un archivo CSV
    
    
    return results
    
#------------------------------------------------------------------------------------------------------------

def extract_feature_vectors(device, autoencoder, data_loader):
    
    output_dir = '../vectors'
    os.makedirs(output_dir, exist_ok=True)  # Asegúrate de que el directorio exista

    with torch.no_grad():
        autoencoder.eval()

        for images, img_names in tqdm(data_loader, desc="Extrayendo características"):
            autoencoder.extraer_caracteristicas(device, images, img_names, output_dir)

    print(f"Feature vectors saved in '{output_dir}'")

#------------------------------------------------------------------------------------------------------------

def Ejecutar_AutoEncoder(train_and_extract=True, mostrar_img=False):
    
    torch.cuda.empty_cache()
    
    if not os.path.exists("../pesos"):
        os.makedirs("../pesos")
    
    config = configparser.ConfigParser()
    # Leemos el archivo de configuracion
    config.read('./config.cfg')
    
    train_dir = config['Carpetas']['carpeta_recortes_train']
    val_dir = config['Carpetas']['carpeta_recortes_val']
    test_dir = config['Carpetas']['carpeta_recortes_test']
    
    num_kernels_str = config['Parametros']['num_kernels']
    num_kernels = ast.literal_eval(num_kernels_str)
    
    depths_str = config['Parametros']['depths']  
    depths = ast.literal_eval(depths_str) 
    
    learning_rates_str = config['Parametros']['learning_rates']
    learning_rates = ast.literal_eval(learning_rates_str)
    
    batch_size = config['Parametros']['batch_size']
    batch_size = int(batch_size)
    
    num_epochs = config['Parametros']['num_epochs']
    num_epochs = int(num_epochs)

    
    if train_and_extract:
        param_grid = {
            'num_kernels': num_kernels,
            'depth': depths,
            'learning_rate': learning_rates
        }
        print(param_grid)
        results = grid_search(train_dir, 
                              val_dir, 
                              test_dir, 
                              param_grid, 
                              num_epochs=num_epochs,
                              batch_size=batch_size)
        
        Ejecutar_AutoEncoder(train_and_extract = False)
            
    else:
        device = Elegir_Dispositivo(solo_CPU=False)
        output_dir = '../vectors'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
                
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        
        train_dataset = CustomImageDataset(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = CustomImageDataset(val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = CustomImageDataset(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        if mostrar_img:
            display_best_average_worst_models("resultados_autoencoder.json", device=device, data_loader=test_loader, num_images=1)
            
        # En este caso, cargamos un modelo ya entrenado desde los pesos guardados
        best_autoencoder = Autoencoder().to(device)
        
        # Si tienes un archivo específico para cargar, lo harías aquí
        
        best_autoencoder, _, aux = choose_best_model('./resultados_autoencoder.json', device)
        display(aux)
        del aux
        gc.collect()
        best_autoencoder.eval()
        
        extract_feature_vectors(device, best_autoencoder, train_loader)
        extract_feature_vectors(device, best_autoencoder, val_loader)
        extract_feature_vectors(device, best_autoencoder, test_loader)

#------------------------------------------------------------------------------------------------------------

