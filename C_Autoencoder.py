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

#------------------------------------------------------------------------------------------------------------

def Error_Ponderado(MSE, SSIM, alpha):
    return (SSIM * alpha) + (MSE * (1 - alpha))

#------------------------------------------------------------------------------------------------------------

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim_value = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_value.item()

#------------------------------------------------------------------------------------------------------------

def Elegir_Dispositivo(solo_CPU=True):
    if solo_CPU:
        device = torch.device("cpu")
    else:
        num_gpus = torch.cuda.device_count()
        max_memory_gpu = None
        max_free_memory = 0
        for i in range(num_gpus):
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            print(f'GPU {i}: Memoria libre: {free_memory / 1024**3:.2f} GB de {total_memory / 1024**3:.2f} GB totales')
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                max_memory_gpu = i
        device = torch.device(f'cuda:{max_memory_gpu}') if max_memory_gpu is not None else torch.device("cpu")
        torch.cuda.set_device(device)
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

class Autoencoder(nn.Module):
    def __init__(self, num_kernels=4, depth=6, bottleneck_channels=64):
        super(Autoencoder, self).__init__()
        layers = []
        in_channels = 3
        for i in range(depth - 1):
            out_channels = num_kernels * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        layers = []
        in_channels = bottleneck_channels
        for i in range(depth - 1):
            out_channels = num_kernels * (2 ** (depth - i - 2))
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return torch.clamp(decoded, 0, 255)

    def encode(self, x):
        return self.encoder(x)

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

def store_results(results, file_path="results.json"):
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

#------------------------------------------------------------------------------------------------------------

def choose_best_model(results_path="results.json", device=None):
    if not os.path.exists(results_path):
        print(f"Error: El archivo de resultados {results_path} no existe.")
        return None, None, None

    # Cargar los resultados guardados en el JSON
    with open(results_path, "r") as f:
        results = json.load(f)

    # Seleccionar el mejor modelo basado en la pérdida de validación
    best_model = min(results, key=lambda x: results[x]["validation_loss"])
    best_params = results[best_model]["params"]
    best_val_loss = results[best_model]["validation_loss"]

    print(f"Mejor modelo encontrado: {best_model}")
    print(f"Parámetros: {best_params}")
    print(f"Pérdida de Validación: {best_val_loss:.4f}")

    # Cargar los pesos del modelo entrenado con la configuración del mejor modelo
    param_str = f"nk{best_params['num_kernels']}_d{best_params['depth']}_lr{best_params['learning_rate']}"
    weight_path = f"./pesos/autoencoder_{param_str}.pth"

    if not os.path.exists(weight_path):
        print(f"Error: no se encontraron los pesos en {weight_path}")
        return None, None, None

    # Crear el autoencoder con los mejores parámetros y cargar los pesos
    best_autoencoder = Autoencoder(num_kernels=best_params['num_kernels'], depth=best_params['depth']).to(device)
    # Load only the state_dict (model weights)
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    
    # Load the state_dict into the model
    best_autoencoder.load_state_dict(state_dict)
    best_autoencoder.eval()

    print(f"Pesos cargados desde: {weight_path}")

    return best_autoencoder, best_params, best_val_loss


#------------------------------------------------------------------------------------------------------------

def train_autoencoder(device, autoencoder, train_loader, val_loader, num_epochs=20, learning_rate=0.001, param_dict=None):
    autoencoder.train()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    
    # Extraer los hiperparámetros en un string para el nombre del archivo
    param_str = f"nk{param_dict['num_kernels']}_d{param_dict['depth']}_lr{learning_rate}"
    
    for epoch in range(num_epochs):
        alpha = max(0, 1 - (epoch / (num_epochs - 1)))  # alpha va de 1 a 0
        train_loss = 0.0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            
            mse_loss = criterion_mse(outputs, images)
            ssim_loss = 1 - ssim(outputs, images)
            
            ssim_loss = torch.tensor(ssim_loss, device=device)
            weighted_loss = Error_Ponderado(mse_loss, ssim_loss, alpha)
            
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        val_loss = 0.0
        autoencoder.eval()
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = autoencoder(images)
                mse_loss = criterion_mse(outputs, images)
                ssim_loss = 1 - ssim(outputs, images)
                
                ssim_loss = torch.tensor(ssim_loss, device=device)
                weighted_loss = Error_Ponderado(mse_loss, ssim_loss, alpha)
                val_loss += weighted_loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        autoencoder.train()
    
    # Guardar los pesos del modelo con el nombre que incluye los hiperparámetros al finalizar el entrenamiento
    weight_path = f"./pesos/autoencoder_{param_str}.pth"
    torch.save(autoencoder.state_dict(), weight_path)
    print(f"Pesos guardados en: {weight_path}")
    
    return train_losses, val_losses


#------------------------------------------------------------------------------------------------------------

def grid_search(train_dir, val_dir, test_dir, device, param_grid, num_epochs=20, batch_size=32):
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    train_dataset = CustomImageDataset(train_dir, transform=transform)
    val_dataset = CustomImageDataset(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Probando configuración: {param_dict}")
        
        autoencoder = Autoencoder(num_kernels=param_dict['num_kernels'], depth=param_dict['depth']).to(device)
        train_losses, val_losses = train_autoencoder(device, autoencoder, train_loader, val_loader, num_epochs=num_epochs, learning_rate=param_dict['learning_rate'], param_dict=param_dict)
        
        results[str(param_dict)] = {
            "params": param_dict,
            "training_loss": train_losses,
            "validation_loss": val_losses[-1]
        }
        print(f"Configuración {param_dict}, Última Validation Loss: {val_losses[-1]:.4f}")
        
        del autoencoder
        torch.cuda.empty_cache()
        clear_output()
    
    store_results(results)
    return results

#------------------------------------------------------------------------------------------------------------

def extract_feature_vectors(device, autoencoder, data_loader):
    # Ensure the output directory exists
    output_dir = './vectors'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    autoencoder.eval()
    patient_vectors = defaultdict(list)

    with torch.no_grad():
        for images, img_names in tqdm(data_loader, desc="Extrayendo características"):
            images = images.to(device)
            encoded_features = autoencoder.encode(images).cpu().numpy()
            
            for feature_vector, img_name in zip(encoded_features, img_names):
                # Use raw string to avoid escape sequence issues
                patient_id = re.split(r"_|\.|/|\\", img_name)[0]
                patient_vectors[patient_id].append(feature_vector.flatten().tolist())

    # Calculate and save aggregated feature vectors for each patient
    for patient_id, vectors in patient_vectors.items():
        vectors_array = np.array(vectors)
        
        # Compute the five operations
        operations = {
            "maximum": np.max(vectors_array, axis=0),
            "minimum": np.min(vectors_array, axis=0),
            "mean": np.mean(vectors_array, axis=0),
            "sum": np.sum(vectors_array, axis=0),
            "median": np.median(vectors_array, axis=0)
        }
        
        # Convert to DataFrame with each operation as a row
        df = pd.DataFrame(operations)
        
        # Save to a CSV file with the patient_id as the file name
        csv_path = os.path.join(output_dir, f"{patient_id}.csv")
        df.to_csv(csv_path, index=False)
    
    print(f"Feature vectors with aggregated operations saved in '{output_dir}'")

#------------------------------------------------------------------------------------------------------------

def save_feature_vectors(feature_vectors, file_path="feature_vectors.json"):
    with open(file_path, "w") as f:
        json.dump(feature_vectors, f, indent=4)

#------------------------------------------------------------------------------------------------------------

def Ejecutar_AutoEncoder(train_and_extract=True):

    if not os.path.exists("./pesos"):
        os.makedirs("./pesos")
    
    device = Elegir_Dispositivo(solo_CPU=False)
    
    config_path = "config.cfg"
    
    train_dir, val_dir, test_dir, num_kernels, depths, learning_rates, batch_size, num_epochs = load_config(config_path)
    
    if train_and_extract:
        param_grid = {
            'num_kernels': num_kernels,
            'depth': depths,
            'learning_rate': learning_rates
        }
        
        results = grid_search(train_dir, val_dir, test_dir, device, param_grid, num_epochs=num_epochs, batch_size=batch_size)
        
        best_autoencoder, best_params, best_val_loss = choose_best_model('./results.json', device)
        
        if best_autoencoder is None:
            return None, None
        
        # Realizamos la extracción de características con el mejor autoencoder cargado
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        
        train_dataset = CustomImageDataset(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = CustomImageDataset(val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = CustomImageDataset(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        extract_feature_vectors(device, best_autoencoder, train_loader)
        extract_feature_vectors(device, best_autoencoder, val_loader)
        extract_feature_vectors(device, best_autoencoder, test_loader)
            
    else:
        # En este caso, cargamos un modelo ya entrenado desde los pesos guardados
        best_autoencoder = Autoencoder().to(device)
        
        # Si tienes un archivo específico para cargar, lo harías aquí
        best_autoencoder, _, _ = choose_best_model('./results.json', device)
        
        best_autoencoder.eval()
        
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        
        train_dataset = CustomImageDataset(train_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = CustomImageDataset(val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = CustomImageDataset(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        extract_feature_vectors(device, best_autoencoder, train_loader)
        extract_feature_vectors(device, best_autoencoder, val_loader)
        extract_feature_vectors(device, best_autoencoder, test_loader)

#------------------------------------------------------------------------------------------------------------
