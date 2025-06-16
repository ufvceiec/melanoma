import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim_value = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_value.item()

def Error_Ponderado(MSE, SSIM, alpha):
    return ((MSE * alpha) + (SSIM * (1 - alpha)))

def Execute_epoch(autoencoder, train_loader, val_loader, optimizer, criterion_mse, device, num_epochs, epoch, results_train):
    train_losses = []
    val_losses = []
    
    scaler = torch.amp.GradScaler(device='cuda')

    autoencoder.train()
    train_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
    
    for images, _ in progress_bar:
        images = images.to(device)
        
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
        
            outputs = autoencoder.forward(images)
            
            mse_loss = criterion_mse(outputs, images)
        
            if (epoch > num_epochs * 0.8) & (num_epochs > 1):
                
                alpha = min(1, ( -(epoch + 1) + num_epochs ) / ( 0.2 * num_epochs ) )  # alpha va de 1 a 0
                
                ssim_loss = 1 - ssim(outputs, images)
                
                ssim_loss = torch.tensor(ssim_loss, device=device)
            else:
                alpha = 1
                
                ssim_loss = 0

        
        weighted_loss = Error_Ponderado(mse_loss, ssim_loss, alpha)
        
        optimizer.zero_grad()
        
        scaler.scale(weighted_loss).backward()
        
        scaler.step(optimizer)
        
        scaler.update()
        
        train_loss += weighted_loss.item()
        
        progress_bar.set_postfix(train_loss=train_loss / (progress_bar.n + 1))  # Promedio actual de train_loss
        
        del images

       
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
        
    print(f"MSE = {mse_loss}")
    print(f"SSIM = {ssim_loss}")
    print(f"Alpha = {alpha}")
    
    val_loss = 0.0
    autoencoder.eval()
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc=f"Validation epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            outputs = autoencoder.forward(images)
            
            mse_loss = criterion_mse(outputs, images)
            
            if (epoch > num_epochs * 0.8) & (num_epochs > 1):
                
                alpha = min(1, ( -(epoch + 1) + num_epochs ) / ( 0.2 * num_epochs ) )  # alpha va de 1 a 0
                
                ssim_loss = 1 - ssim(outputs, images)
                
                ssim_loss = torch.tensor(ssim_loss, device=device)
            else:
                alpha = 1
                
                ssim_loss = 0
            
            weighted_loss = Error_Ponderado(mse_loss, ssim_loss, alpha)
            
            val_loss += weighted_loss.item()

            del images
            
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"MSE = {mse_loss}")
        print(f"SSIM = {ssim_loss}")      
        print(f"Alpha = {alpha}")
        
        results_train = {
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Alpha = {alpha} Training Loss: {train_loss}, Validation Loss: {val_loss}")
    
    autoencoder.train()
    
    torch.cuda.empty_cache()
    
    return results_train