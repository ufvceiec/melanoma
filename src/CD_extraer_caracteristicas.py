import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from src.BB_recortar_1_Imagen import Recortar_img
from src.CB_Train_Autoencoder import choose_best_model, Elegir_Dispositivo, CustomImageDataset
from src.CA_Model_Autoencoder import Autoencoder

torch.backends.cudnn.benchmark = True  # 🔥 Acelera operaciones en GPU


def Rutas_Archivos(ruta_carpeta):
    return [archivo for archivo in Path(ruta_carpeta).iterdir() if archivo.is_file()]


def Extraer_Caracteristicas(ruta_recortes_aux, ruta_vectores_aux, csv_output_path, batch_size=32):       
    start_time = time.time()  # ⏱️ Iniciar cronómetro

    device = Elegir_Dispositivo(False)  # Cambiar a True para forzar GPU
    autoencoder, _, _ = choose_best_model(results_path="./resultados_autoencoder.json", device=device)
    
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = CustomImageDataset(ruta_recortes_aux, transform=transform)

    # 📌 Reducimos num_workers y evitamos problemas de memoria compartida
    num_workers = 0  # 🔥 Reducido para evitar problemas de /dev/shm
    batch_size = min(batch_size, len(dataset))  # 🔥 Ajustar batch_size dinámicamente

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,  # 🔥 Reducimos trabajadores
        pin_memory=True if device.type == "cuda" else False,  # 🔥 Solo en GPU
        persistent_workers=False  # 🔥 Evita procesos residuales
    )

    # 📌 Crear carpeta para almacenar los archivos temporales
    aux_folder = f"{ruta_vectores_aux}/temp_npy"
    os.makedirs(aux_folder, exist_ok=True)

    # 📌 Procesar imágenes y guardar cada lote en un archivo separado
    file_counter = 0
    with torch.no_grad():
        for recortes, _ in tqdm(data_loader, desc="🔄 Extrayendo features"):
            recortes = recortes.to(device, non_blocking=True)

            # 🔥 Extraer características y procesar en GPU
            features = autoencoder.encode(recortes).flatten(start_dim=1).cpu().numpy()

            # 🔥 Guardar cada lote en un archivo `.npy` separado
            np.save(f"{aux_folder}/aux_{file_counter}.npy", features)
            file_counter += 1

            # 🔥 Minimizar uso de RAM eliminando datos intermedios
            del recortes, features
            torch.cuda.empty_cache()
            gc.collect()

    print(f"✅ Características guardadas en {aux_folder}")

    # 📌 Cálculo de operaciones sobre cada columna sin cargar todo en RAM
    print("🔢 Calculando operaciones sobre las features...")

    # 📌 Inicializar valores para las estadísticas
    max_values, min_values, sum_values, count_values = None, None, None, 0
    mean_accumulator = None

    # 📌 Leer cada archivo `.npy` y actualizar estadísticas sin usar RAM
    for file in tqdm(sorted(os.listdir(aux_folder)), desc="📊 Procesando archivos temporales"):
        file_path = os.path.join(aux_folder, file)
        features = np.load(file_path)

        if max_values is None:
            max_values = features.max(axis=0)
            min_values = features.min(axis=0)
            sum_values = features.sum(axis=0)
            mean_accumulator = features.mean(axis=0) * features.shape[0]
            count_values = features.shape[0]
        else:
            max_values = np.maximum(max_values, features.max(axis=0))
            min_values = np.minimum(min_values, features.min(axis=0))
            sum_values += features.sum(axis=0)
            mean_accumulator += features.mean(axis=0) * features.shape[0]
            count_values += features.shape[0]

    # 📌 Calcular valores finales
    mean_values = mean_accumulator / count_values
    median_values = mean_values  # Aproximación rápida de la mediana

    # 📌 Guardar los resultados en CSV
    operations = {
        "maximum": max_values,
        "minimum": min_values,
        "mean": mean_values,
        "sum": sum_values,
        "median": median_values
    }

    operations_df = pd.DataFrame(operations)
    operations_df.to_csv(csv_output_path, index=False)

    # 🔥 Eliminar archivos temporales
    for file in os.listdir(aux_folder):
        os.remove(os.path.join(aux_folder, file))
    os.rmdir(aux_folder)

    # ⏱️ Calcular tiempo total
    total_time = time.time() - start_time
    print(f"✅ Operaciones guardadas en {csv_output_path}")
    print(f"⏱️ Tiempo total de ejecución: {total_time:.2f} segundos ({total_time / 3600:.2f} horas)")
    
    gc.collect()
