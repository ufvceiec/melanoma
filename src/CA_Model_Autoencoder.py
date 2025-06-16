import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from src.CC_Autoencoder_Exec import Execute_epoch

#------------------------------------------------------------------------------------------------------------

def Error_Ponderado(MSE, SSIM, alpha):
    return (MSE * alpha) + (SSIM * (1 - alpha))

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

torch.backends.cudnn.benchmark = True  # Selección automática de kernels óptimos

class Autoencoder(nn.Module):
    def __init__(self, num_kernels=4, depth=6):
        super(Autoencoder, self).__init__()

        # Encoder
        layers = []
        in_channels = 3
        for i in range(depth):
            out_channels = num_kernels * (2 ** i)
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=2))  # Convolution 5x5, stride=2
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1))  # Convolution 3x3, stride=2
            layers.append(nn.ReLU(inplace=True))  # Usar inplace=True para ahorrar memoria
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        for i in range(depth):
            out_channels = num_kernels * (2 ** (depth - i - 1))
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0))
            layers.append(nn.ReLU(inplace=True))  # Usar inplace=True para ahorrar memoria
            in_channels = out_channels

        # Última capa para reconstrucción
        layers.append(nn.Conv2d(in_channels, 3, kernel_size=5, padding=2))  # Reconstrucción final
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        original_size = x.size()[-2:]  # Guardar el tamaño original de entrada (H, W)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Interpolar la salida decodificada al tamaño original
        decoded = F.interpolate(decoded, size=original_size, mode='bilinear', align_corners=False)
        return torch.clamp(decoded, 0.0, 1.0)

    def encode(self, x):
        return self.encoder(x)

    def extraer_caracteristicas(self, device, images, img_names, output_dir, batch_size=64):
        """
        Extrae características latentes de un conjunto de imágenes en lotes para mayor eficiencia.
        """
        # Asegurarnos de que las imágenes están en el dispositivo correcto
        images = images.to(device)
        dataset_size = images.size(0)

        # Procesar por lotes
        with torch.no_grad():
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch = images[start:end]
                encoded_features = self.encode(batch).cpu().numpy()

                # Guardar cada vector de características
                for feature_vector, img_name in zip(encoded_features, img_names[start:end]):
                    np.save(os.path.join(output_dir, f"{img_name}.npy"), feature_vector.flatten())

        # Liberar memoria inmediatamente
        del images
        torch.cuda.empty_cache()
        
    def extraer_caracteristicas_img(self, device, imagen, img_name, output_dir):
        """
            Extrae características latentes de un conjunto de imágenes en lotes para mayor eficiencia.
        """
        # Asegurarnos de que las imágenes están en el dispositivo correcto
        imagen = imagen.to(device)
    
        # Procesar por lotes
        with torch.no_grad():
            encoded_features = self.encode(imagen)
    
            # Mover las características codificadas a la CPU antes de convertirlas a NumPy
            encoded_features = encoded_features.cpu()
    
            # Guardar cada vector de características
            #np.save(os.path.join(output_dir, f"{img_name}.npy"), encoded_features.flatten())
    
        # Liberar memoria inmediatamente
        del imagen
        torch.cuda.empty_cache()
        return encoded_features.flatten()

#------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

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

def train_autoencoder(device, 
                      autoencoder, 
                      train_loader, 
                      val_loader, 
                      test_loader, 
                      num_epochs=20, 
                      learning_rate=0.001, 
                      param_dict=None):

    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    test_losses = []  # Lista para almacenar las pérdidas del conjunto de prueba

    results_train = {}
    results_test = {}
    param_str = f"nk{param_dict['num_kernels']}_d{param_dict['depth']}_lr{learning_rate}"
    
    for epoca in range(num_epochs):
        results_train = Execute_epoch(autoencoder, train_loader, val_loader, optimizer, criterion_mse, device, num_epochs, epoca, results_train)

    mostrar_imagenes_reconstruidas(autoencoder, test_loader, device, num_imagenes=3)
    
    autoencoder.eval()

    test_losses_MSE = []
    test_losses_SSIM = []
    
    test_loss_MSE = 0.0
    test_loss_SSIM = 0.0
    with torch.no_grad():
        
        for images, _ in tqdm(test_loader, desc=f"Test {param_str}"):

            images = images.to(device)
            
            outputs = autoencoder(images)
            
            mse_loss = criterion_mse(outputs, images)
            
            ssim_loss = 1 - ssim(outputs, images)
            
            ssim_loss = torch.tensor(ssim_loss, device=device)
            
            test_loss_MSE += mse_loss
            
            test_loss_SSIM += ssim_loss

            del images

    
    test_loss_MSE /= len(test_loader)
    test_losses_MSE.append(test_loss_MSE)

    test_loss_SSIM /= len(test_loader)
    test_losses_SSIM.append(test_loss_SSIM)

    # Guardar los pesos del modelo con el nombre que incluye los hiperparámetros al finalizar el entrenamiento
    weight_path = f"../pesos/autoencoder_{param_str}.pth"
    torch.save(autoencoder.state_dict(), weight_path)
    print(f"Pesos guardados en: {weight_path}")
    
    # Guardar los resultados (entrenamiento, validación, prueba)
    results_test = {
        "test_loss_MSE": test_losses_MSE,
        "test_loss_SSIM": test_losses_SSIM
    }
    torch.cuda.empty_cache()
    return results_train, results_test