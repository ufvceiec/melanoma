import os
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

import itertools
from itertools import product

from collections import defaultdict, Counter

import matplotlib.pyplot as plt

import seaborn as sns
import gc

from scipy.stats import mode

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from scipy.stats import mode
from joblib import Parallel, delayed
import numpy as np
import torch
from typing import Union, Optional, Tuple

from pathlib import Path

from graphviz import Source
from sklearn.tree import export_graphviz

#------------------------------------------------------------------------------

def guardar_vectores_mal_clasificados(y_test, y_pred, file_names, vector_type, model_name):
    """
    Guarda un archivo CSV con los nombres de los vectores mal clasificados y sus etiquetas.
    """
    import os
    import pandas as pd
    import numpy as np

    # Identificar errores
    errores_idx = np.where(y_test != y_pred)[0]

    if len(errores_idx) == 0:
        print(f"✅ No se encontraron vectores mal clasificados para {model_name} - {vector_type}.")
        return

    # Obtener nombres e info de errores
    nombres_erroneos = [file_names[i] if file_names else f"vector_{i}" for i in errores_idx]
    etiquetas_reales = y_test[errores_idx]
    etiquetas_predichas = y_pred[errores_idx]

    # Crear DataFrame
    errores_df = pd.DataFrame({
        "Nombre_Archivo": nombres_erroneos,
        "Etiqueta_Real": etiquetas_reales,
        "Etiqueta_Predicha": etiquetas_predichas
    })

    # Guardar CSV
    output_dir = "./vectores_mal_clasificados"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_{vector_type}_errores.csv")
    errores_df.to_csv(csv_path, index=False)

    print(f"❗️ Se guardaron {len(errores_idx)} errores en '{csv_path}'")

#------------------------------------------------------------------------------

def Elegir_Dispositivo(solo_CPU=True, verborrea=True):
    if solo_CPU:
        device = torch.device("cpu")
    else:
        num_gpus = torch.cuda.device_count()
        max_memory_gpu = None
        max_free_memory = 0
        for i in range(num_gpus):
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            #print(f'GPU {i}: Memoria libre: {free_memory / 1024**3:.2f} GB de {total_memory / 1024**3:.2f} GB totales')
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                max_memory_gpu = i
        if max_memory_gpu is not None:
            device = torch.device(f'cuda:{max_memory_gpu}')
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    if verborrea:
        print(f"Usando dispositivo: {device}")
    return device
    
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self):
        self.epochs = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

    def update(self, epoch, train_acc, val_acc, train_loss=None, val_loss=None):
        self.epochs.append(epoch)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)

    def plot(self, title, vector_type):
        plt.figure(figsize=(12, 6))
        plt.plot(self.epochs, self.train_acc, label='Train Accuracy')
        plt.plot(self.epochs, self.val_acc, label='Validation Accuracy')

        plt.title(f"{title} - {vector_type}")
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_plot(self, title, vector_type, save_path):
        plt.figure(figsize=(12, 6))

        # Gráfico de Precisión
        plt.plot(self.epochs, self.train_acc, label='Train Accuracy')
        plt.plot(self.epochs, self.val_acc, label='Validation Accuracy')

        # Configuración adicional del gráfico
        plt.title(f"{title} - {vector_type}")
        plt.xlabel('Épocas')
        plt.ylabel('Métrica')
        plt.legend()
        plt.grid(True)

        # Guardar el gráfico
        plt.savefig(save_path)
        plt.close()  # Cierra el gráfico para liberar memoria
        print(f"📈 Gráfico guardado en '{save_path}'")

#------------------------------------------------------------------------------

class KNN:
    def __init__(self, n_neighbors=5, weights='uniform'):
        if weights not in {'uniform', 'distance'}:
            raise ValueError("El parámetro 'weights' solo puede ser 'uniform' o 'distance'.")
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _convert_to_tensor(self, data, dtype=torch.float32):
        """Convierte datos en tensores si es necesario."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return torch.tensor(data.values, dtype=dtype)
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype)
        return data

    def fit(self, X, y, X_val=None, y_val=None, fold=0):
        X = self._convert_to_tensor(X)
        y = self._convert_to_tensor(y, dtype=torch.long)

        # 🚨 Validación para ajustar el número de vecinos
        n_samples = X.shape[0]
        if n_samples < self.n_neighbors:
            print(f"⚠️ Advertencia: El conjunto de entrenamiento tiene solo {n_samples} muestras. "
                  f"Reduciendo el número de vecinos a {n_samples}.")
            self.n_neighbors = n_samples

        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if X is None:
            raise ValueError("Error: El conjunto de datos de entrada (X) es None. Verifica que se haya cargado correctamente.")

        X = self._convert_to_tensor(X)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # 🚨 Asegurar que el número de vecinos no supere el número de muestras
        num_neighbors = min(self.n_neighbors, self.X_train.shape[0])

        # Cálculo de distancias optimizado
        distances = torch.cdist(X, self.X_train)
        knn_indices = distances.topk(num_neighbors, largest=False).indices
        knn_labels = self.y_train[knn_indices]

        if self.weights == 'uniform':
            predictions = torch.mode(knn_labels, dim=1).values
        else:  # weights == 'distance'
            knn_distances = distances.gather(1, knn_indices)
            weights = 1 / (knn_distances + 1e-8)

            weighted_votes = torch.zeros((X.shape[0], self.y_train.max().item() + 1))
            weighted_votes.scatter_add_(1, knn_labels, weights)

            predictions = torch.argmax(weighted_votes, dim=1)

        return predictions.numpy()
    def obtener_vecinos_mal_clasificados(self, model, X_test, y_test, file_names_test, file_names_train):
        """
        Muestra los vecinos más cercanos de cada vector mal clasificado
        y devuelve la lista de vectores mal clasificados en formato compatible para visualización.
        """
        import numpy as np
        import torch
    
        # Asegurarse de que los datos estén en formato numpy
        y_test = np.array(y_test)
        y_pred = model.predict(X_test)
    
        errores_idx = np.where(y_test != y_pred)[0]
        print(f"🔍 Se encontraron {len(errores_idx)} vectores mal clasificados.")
    
        vectores_erroneos = []
    
        # Verificación de consistencia
        if len(model.X_train) != len(file_names_train):
            print(f"❌ Tamaño inconsistente: X_train = {len(model.X_train)}, nombres = {len(file_names_train)}")
            print("⚠️ Asegúrate de que 'file_names_train' corresponde exactamente con los datos usados para entrenamiento.")
            return []
    
        for idx in errores_idx:
            # Convertir a tensor si es necesario y asegurarse de que esté en la misma device que X_train
            if isinstance(X_test, torch.Tensor):
                x = X_test[idx].unsqueeze(0)
            else:
                x = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)
    
            if isinstance(model.X_train, np.ndarray):
                model_X_train_tensor = torch.tensor(model.X_train, dtype=torch.float32)
            else:
                model_X_train_tensor = model.X_train
    
            model_X_train_tensor = model_X_train_tensor.to(x.device)
    
            # Calcular distancias
            distances = torch.cdist(x, model_X_train_tensor)
            knn_indices = distances.topk(min(self.n_neighbors, model_X_train_tensor.shape[0]), largest=False).indices.squeeze().cpu().numpy()
    
            nombre_archivo = file_names_test[idx]
            etiqueta_real = y_test[idx]
    
            print(f"\n🔴 Vector mal clasificado: {nombre_archivo} (Real: {etiqueta_real}, Predicho: {y_pred[idx]})")
            print("📎 Vecinos más cercanos:")
    
            for rank, i in enumerate(knn_indices[:self.n_neighbors]):
                if i >= len(file_names_train):
                    print(f"⚠️ Índice fuera de rango: {i} — nombres disponibles: {len(file_names_train)}")
                    continue
    
                nombre_vecino = file_names_train[i]
                label_vecino = (
                    model.y_train[i].item() if isinstance(model.y_train[i], torch.Tensor) else model.y_train[i]
                )
    
                print(f"   {rank + 1}. Index: {i}, Nombre: {nombre_vecino}, Etiqueta: {label_vecino}")
    
            # Guardar vector mal clasificado para visualización
            vector_np = (
                X_test[idx].cpu().numpy() if isinstance(X_test, torch.Tensor) else np.array(X_test[idx])
            )
            vectores_erroneos.append((vector_np, etiqueta_real, nombre_archivo))

        return vectores_erroneos


    


#------------------------------------------------------------------------------

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils import resample
from joblib import Parallel, delayed
from scipy.stats import mode
import torch

class RandomForestOptimized:
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 split_criterion='gini',
                 min_samples_split=2,
                 n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.trees = []
        self.classes_ = None
        self.feature_names = None

    def _convert_to_numpy(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if y is not None and isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
            return X, y
        return X

    def _build_tree(self, X, y, seed):
        X_sample, y_sample = resample(X, y, replace=True, random_state=seed)
        tree = DecisionTreeClassifier(
            criterion=self.split_criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        tree.fit(X_sample, y_sample)
        return tree

    def fit(self, X, y, X_val=None, y_val=None, fold=0):
        X, y = self._convert_to_numpy(X, y)
        self.classes_ = np.unique(y)

        # Guardar nombres de características si existen (por ejemplo, con DataFrames)
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'f{i}' for i in range(X.shape[1])]

        seeds = np.random.randint(0, 10000, self.n_estimators)
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y, seed) for seed in seeds
        )

    def predict(self, X):
        X = self._convert_to_numpy(X)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return mode(predictions, axis=0, keepdims=False).mode

    def predict_proba(self, X):
        X = self._convert_to_numpy(X)
        n_classes = len(self.classes_)
        probas = np.zeros((X.shape[0], n_classes))

        for tree in self.trees:
            tree_proba = tree.predict_proba(X)
            if tree_proba.shape[1] != n_classes:
                temp = np.zeros_like(probas)
                class_idx = {cls: i for i, cls in enumerate(self.classes_)}
                for i, cls in enumerate(tree.classes_):
                    temp[:, class_idx[cls]] = tree_proba[:, i]
                probas += temp
            else:
                probas += tree_proba

        return probas / self.n_estimators

    def feature_importances(self):
        """Promedia la importancia de las variables de todos los árboles."""
        importances = np.array([tree.feature_importances_ for tree in self.trees])
        return np.mean(importances, axis=0)

    def vote_distribution(self, instance):
        """Devuelve cuántos árboles votaron por cada clase para una instancia específica."""
        instance = instance.reshape(1, -1)
        votes = np.array([tree.predict(instance)[0] for tree in self.trees])
        unique, counts = np.unique(votes, return_counts=True)
        return dict(zip(unique, counts))

    def print_tree(self, index=0):
        """Muestra la estructura de un árbol específico."""
        if 0 <= index < len(self.trees):
            print(export_text(self.trees[index], feature_names=self.feature_names))
        else:
            print("Índice de árbol fuera de rango.")
    def export_tree_graph(self, tree_index=0, max_depth=3, show_plot=True, save_path=None):
        """
        Exporta y visualiza un árbol individual del bosque.
        
        Args:
            tree_index (int): Índice del árbol a exportar.
            max_depth (int): Profundidad máxima mostrada del árbol.
            show_plot (bool): Si True, muestra el gráfico.
            save_path (str): Si no es None, guarda el gráfico en el path especificado.
        """
        if tree_index >= len(self.trees) or tree_index < 0:
            print(f"Índice de árbol fuera de rango. Hay {len(self.trees)} árboles en total.")
            return
        
        tree = self.trees[tree_index]

        dot_data = export_graphviz(
            tree,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=max_depth,
            proportion=True,
            impurity=False
        )
        
        graph = Source(dot_data)
        
        if show_plot:
            display(graph)
        
        if save_path is not None:
            graph.render(save_path, format='png', cleanup=True)
            graph.render(save_path, format='pdf', cleanup=True)
            print(f"📄 Árbol guardado en: {save_path}.png")

    def plot_forest_structure(self, max_depth=10):
        """
        Representa las hojas del Random Forest:
        - Eje Y: profundidad del nodo hoja (más profundo, más abajo).
        - Eje X: posición visual arbitraria para separación.
        - Tamaño del punto: proporcional logarítmicamente al número de muestras.
        - Color: clase que predice el nodo hoja.
        """
        if not self.trees:
            print("Primero entrena el modelo con .fit().")
            return
    
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
    
        x_vals = []
        y_vals = []
        sizes = []
        colors = []
    
        cmap = get_cmap("tab10")
        n_classes = len(self.classes_)
    
        counter = 0  # Para el eje X visual
    
        for tree in self.trees:
            tree_ = tree.tree_
            values = tree_.value.squeeze(axis=1)
            samples = tree_.n_node_samples
    
            def collect_leaf_data(node_id, depth):
                nonlocal counter
                is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
                if depth > max_depth:
                    return
                if is_leaf:
                    class_id = np.argmax(values[node_id])
                    count = samples[node_id]
                    x_vals.append(counter)
                    y_vals.append(depth)
                    sizes.append(np.log1p(count) * 30)
                    colors.append(cmap(class_id % 10))
                    counter += 1
                else:
                    collect_leaf_data(tree_.children_left[node_id], depth + 1)
                    collect_leaf_data(tree_.children_right[node_id], depth + 1)
    
            collect_leaf_data(0, 0)
    
        # Dibujar gráfico
        plt.figure(figsize=(14, 6))
        plt.scatter(x_vals, y_vals, s=sizes, c=colors, edgecolors='k', alpha=0.7)
        plt.gca().invert_yaxis()
        plt.xlabel("Hojas (distribución visual)")
        plt.ylabel("Profundidad del nodo")
        plt.title("Resumen de hojas del Random Forest")
        plt.grid(True)
        plt.show()




#------------------------------------------------------------------------------

from sklearn.svm import SVC

class SVM:
    def __init__(self, input_dim, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel, probability=True)

    def fit(self, X, y, X_val=None, y_val=None, fold=0):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


#------------------------------------------------------------------------------

import torch
from torch import nn, optim
from tqdm import tqdm  # Para visualización de progreso
import numpy as np

class MLPModel:
    def __init__(self, input_dim, hidden_layers, neurons_first_layer, epochs=50, lr=0.001,
                 dropout_rate=0.3, l2_reg=1e-4, patience=5, device="cpu", vector_type=None):
       
        self.device = device
        self.epochs = epochs
        self.vector_type = vector_type
        self.patience = patience

        layers = [
            nn.Linear(input_dim, neurons_first_layer),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ]

        prev_neurons = neurons_first_layer
        for i in range(hidden_layers - 1):
            layers.extend([
                nn.Linear(prev_neurons, prev_neurons // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_neurons //= 2

        layers.extend([nn.Linear(prev_neurons, 1), nn.Sigmoid()])

        self.model = nn.Sequential(*layers).to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_reg)
        self.plotter = TrainingPlotter()
        self.metrics_history = []  # Para almacenar métricas cada 10 épocas

    def _convert_to_tensor(self, data):
        return data.clone().detach().float().to(self.device) if isinstance(data, torch.Tensor) \
               else torch.tensor(data, dtype=torch.float32).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None, fold=-1):
        X, y = self._convert_to_tensor(X), self._convert_to_tensor(y)
        X_val, y_val = (self._convert_to_tensor(X_val), self._convert_to_tensor(y_val)) if X_val is not None else (None, None)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            outputs = self.model(X).squeeze()
            train_loss = self.criterion(outputs, y)
            train_loss.backward()
            self.optimizer.step()

            train_acc = (outputs.round() == y).float().mean().item()

            val_loss, val_acc = 0, 0
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val).squeeze()
                    val_loss = self.criterion(val_outputs, y_val).item()
                    val_acc = (val_outputs.round() == y_val).float().mean().item()

            # Guardar métricas cada 10 épocas
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                metricas_actuales= {
                    "epoch": epoch,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "train_loss": train_loss.item(),
                    "val_loss": val_loss
                }
                #print(metricas_actuales)
                self.metrics_history.append(metricas_actuales)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if fold != -1:
                self.plotter.update(fold * self.epochs + epoch, train_acc, val_acc, train_loss.item(), val_loss)
            
            if patience_counter >= self.patience:    
                break

    def save_metrics_to_csv(self, filepath="mlp_training_metrics.csv"):
        import pandas as pd
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(filepath, index=False)
        print(f"📊 Métricas de entrenamiento guardadas en '{filepath}'")

    def predict(self, X):
        X = self._convert_to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X).squeeze()
            return (outputs > 0.5).int().cpu().numpy()

    def save_weights(self, filepath="mlp_weights.pth"):
        torch.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath="mlp_weights.pth"):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()

    def interpretar_resultados_mlp(self, metrics_path, pesos_path="mlp_weights.pth", show_plot=True):
        if not os.path.exists(metrics_path):
            print(f"❌ No se encontró el archivo de métricas: {metrics_path}")
            return
    
        # Cargar métricas
        df = pd.read_csv(metrics_path)
    
        print("📈 Resumen de métricas del entrenamiento:")
        print(df.describe())
    
        # Gráfico de precisión y pérdida
        if show_plot:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
            # Precisión
            ax[0].plot(df["epoch"], df["train_accuracy"], label="Train Accuracy")
            ax[0].plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")
            ax[0].set_title("Precisión durante el entrenamiento")
            ax[0].set_xlabel("Épocas")
            ax[0].set_ylabel("Accuracy")
            ax[0].legend()
            ax[0].grid(True)

            ax[1].plot(df["epoch"], df["train_loss"], label="Train Loss")
            ax[1].plot(df["epoch"], df["val_loss"], label="Validation Loss")
            ax[1].set_title("Pérdida durante el entrenamiento")
            ax[1].set_xlabel("Épocas")
            ax[1].set_ylabel("Loss")
            ax[1].legend()
            ax[1].grid(True)
        
            plt.tight_layout()
            plt.show()
    
        if pesos_path:
            print(f"📦 Puedes cargar los pesos del modelo desde: {pesos_path}")



#--------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes=2, kernel_size=3, num_filters=32, dropout_rate=0.3, vector_type=None, epochs = 10):
        super(CNN1D, self).__init__()
        self.vector_type = vector_type
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        self.to(self.device)
        self.plotter = TrainingPlotter()

    def forward(self, x):
        x = x.to(self.device).unsqueeze(1)  # Añadir dimensión de canal
        return self.model(x)

    def _convert_to_tensor(self, data, dtype=torch.float32):
        return data.clone().detach().to(self.device) if isinstance(data, torch.Tensor) \
            else torch.tensor(data, dtype=dtype).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None, fold=-1, batch_size=32):
        X, y = self._convert_to_tensor(X), self._convert_to_tensor(y, dtype=torch.long)
        train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val, y_val = self._convert_to_tensor(X_val), self._convert_to_tensor(y_val, dtype=torch.long)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    
        for epoch in range(self.epochs):
            self.train()
            train_loss, train_acc = 0.0, 0.0
    
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
    
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item()
                train_acc += (torch.argmax(outputs, dim=1) == batch_y).float().mean().item()
    
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
    
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        val_outputs = self.forward(batch_X)
                        val_loss += criterion(val_outputs, batch_y).item()
                        val_acc += (torch.argmax(val_outputs, dim=1) == batch_y).float().mean().item()
    
                val_loss /= len(val_loader)
                val_acc /= len(val_loader)
    
            if fold != -1:
                self.plotter.update(fold * self.epochs + epoch, train_acc, val_acc, train_loss, val_loss)


    def predict(self, X, batch_size=32):
        X = self._convert_to_tensor(X)
        dataloader = DataLoader(X, batch_size=batch_size, shuffle=False)
        predictions = []

        self.eval()
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.forward(batch_X)
                predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())

        return np.concatenate(predictions)

    def save_weights(self, filepath="mlp_weights.pth"):
        torch.save(self.model.state_dict(), filepath)

#--------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=4, num_layers=2, dim_feedforward=256,
                 dropout=0.1, positional_encoding=False, num_dense_neurons=100, vector_type=None, epochs = 1):
        super(TransformerClassifier, self).__init__()
        self.epochs = epochs
        self.vector_type = vector_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reduced_dim = min(512, input_dim)
        self.embedding = nn.Linear(input_dim, self.reduced_dim)

        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.pos_encoder = PositionalEncoding(self.reduced_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(self.reduced_dim, num_dense_neurons)
        self.out = nn.Linear(num_dense_neurons, num_classes)
        self.plotter = TrainingPlotter()

        self.to(self.device)

    def forward(self, x):
        x = self.embedding(x.to(self.device)).unsqueeze(1)
        if self.positional_encoding:
            x = self.pos_encoder(x)
        x = self.transformer_encoder(x).mean(dim=1)
        x = torch.relu(self.fc(x))
        return self.out(x)

    def _convert_to_tensor(self, data, dtype=torch.float32):
        return data.clone().detach().to(self.device) if isinstance(data, torch.Tensor) \
            else torch.tensor(data, dtype=dtype).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None, fold=-1, batch_size=32):
        X, y = self._convert_to_tensor(X), self._convert_to_tensor(y, dtype=torch.long)
        train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val, y_val = self._convert_to_tensor(X_val), self._convert_to_tensor(y_val, dtype=torch.long)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        epochs = self.epochs
        for epoch in tqdm(range(epochs), desc=f"Entrenando fold {fold}"):
            self.train()
            train_loss, train_acc = 0.0, 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += (torch.argmax(outputs, dim=1) == batch_y).float().mean().item()

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        val_outputs = self.forward(batch_X)
                        val_loss += criterion(val_outputs, batch_y).item()
                        val_acc += (torch.argmax(val_outputs, dim=1) == batch_y).float().mean().item()

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)

            scheduler.step()
            if fold != -1:
                self.plotter.update(fold * epochs + epoch, train_acc, val_acc, train_loss, val_loss)
        self.plotter.plot("Transformer", self.vector_type)

    def predict(self, X, batch_size=32):
        X = self._convert_to_tensor(X)
        dataloader = DataLoader(X, batch_size=batch_size, shuffle=False)
        predictions = []

        self.eval()
        with torch.no_grad():
            for batch_X in dataloader:
                outputs = self.forward(batch_X)
                predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())

        return np.concatenate(predictions)
        
    def save_weights(self, filepath="mlp_weights.pth"):
        print("Por Hacer")

#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, plot_confusion_matrix=True, model_name='None'):
    # Convertir a NumPy si no lo son
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Validación de entrada
    if len(y_true) != len(y_pred):
        raise ValueError("Las dimensiones de 'y_true' y 'y_pred' no coinciden.")
    
    # Calcular TP, TN, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    total = tp + tn + fp + fn

    # Calcular métricas evitando divisiones por cero
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else -1.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else -1.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0.0  # Se maneja si una clase está ausente

    # Matriz de confusión
    if plot_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.title(f"Matriz de Confusión - {model_name}")
    
        # Guardar la matriz en carpeta de resultados
        results_path = './resultados'
        os.makedirs(results_path, exist_ok=True)
        plot_path = os.path.join(results_path, f"{model_name}_confusion_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Matriz de confusión guardada en '{plot_path}'")

    metrics = {
        "Accuracy": accuracy,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }

    return metrics

#------------------------------------------------------------------------------

def evaluate_model_kfold(model, model_name, X, y, n_splits=5, verbose=False):
    metrics = []
    if n_splits == -1:
        model.fit(X, y)
        y_train_pred = model.predict(X)
        train_metrics = calculate_metrics(y, y_train_pred, plot_confusion_matrix=False)
        metrics.append({
            'train_accuracy': train_metrics['Accuracy'],
            'train_sensitivity': train_metrics['Sensitivity (Recall)'],
            'train_specificity': train_metrics['Specificity'],
            'val_accuracy': 0,
            'val_sensitivity': 0,
            'val_specificity': 0
        })
    else:
        try:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                if len(X_train) == 0 or len(X_val) == 0:
                    print(f"Advertencia: El conjunto de datos está vacío en el fold {fold}. Saltando este fold.")
                    continue

                model.fit(X_train, y_train, X_val=X_val, y_val=y_val, fold=fold)

                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

                train_metrics = calculate_metrics(y_train, y_train_pred, plot_confusion_matrix=False)
                val_metrics = calculate_metrics(y_val, y_val_pred, plot_confusion_matrix=False)

                metrics.append({
                    'train_accuracy': train_metrics['Accuracy'],
                    'train_sensitivity': train_metrics['Sensitivity (Recall)'],
                    'train_specificity': train_metrics['Specificity'],
                    'val_accuracy': val_metrics['Accuracy'],
                    'val_sensitivity': val_metrics['Sensitivity (Recall)'],
                    'val_specificity': val_metrics['Specificity']
                })
        except Exception as e:
            print(f"Error en la división {fold}: {e}")
            return []

    return metrics

#------------------------------------------------------------------------------

def load_data(vectors_dir='./vectores', labels_path='./diagnosticos_corregidos.csv', balance_data=False, limite_por_clase=None):
    import os
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    from tqdm import tqdm

    # Cargar etiquetas
    try:
        labels_df = pd.read_csv(labels_path, sep=';', dtype={'paciente': str})
    except Exception as e:
        raise FileNotFoundError(f"❌ Error al cargar el archivo de etiquetas: {e}")

    labels_dict = dict(zip(labels_df["paciente"], labels_df["diagnostico"]))

    data_by_vector_type = defaultdict(lambda: {0: [], 1: []})
    total_archivos_validos = 0

    # Contador por clase (por tipo de vector)
    contador_por_vector = defaultdict(lambda: {0: 0, 1: 0})

    for filename in tqdm(os.listdir(vectors_dir), desc='Cargando vectores'):
        if not filename.endswith(".csv"):
            continue

        try:
            patient_id = filename.split('.')[0].replace("csv", "").replace("pac", "").strip()
            patient_id = patient_id.split('B')[-1]
            patient_id = str(int(patient_id))  # Formato consistente

            if patient_id not in labels_dict:
                continue

            label = int(labels_dict[patient_id])
            file_path = os.path.join(vectors_dir, filename)
            vector_df = pd.read_csv(file_path)

            required_columns = ["maximum", "minimum", "mean", "sum", "median"]
            if not all(col in vector_df.columns for col in required_columns):
                print(f"❌ Columnas faltantes en {filename}, se omite.")
                continue

            for column in required_columns:
                # Controlar límite por clase
                if limite_por_clase is not None and contador_por_vector[column][label] >= limite_por_clase:
                    continue

                vector = vector_df[column].values.astype(float).reshape(-1)
                vector[np.isinf(vector)] = 0

                data_by_vector_type[column][label].append((vector, label, filename))
                contador_por_vector[column][label] += 1

            total_archivos_validos += 1

        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
            continue

    print(f"\n✅ Archivos válidos procesados: {total_archivos_validos}")
    for vector_type, clases in data_by_vector_type.items():
        print(f"📊 Vector: {vector_type} → Clase 0: {len(clases[0])} | Clase 1: {len(clases[1])}")

    # No hacer balanceo si ya se está forzando un límite específico
    if balance_data and limite_por_clase is None:
        for vector_type in data_by_vector_type:
            min_count = min(len(data_by_vector_type[vector_type][0]),
                            len(data_by_vector_type[vector_type][1]))
            for label in [0, 1]:
                data_by_vector_type[vector_type][label] = data_by_vector_type[vector_type][label][:min_count]
        print("⚖️ Datos balanceados por clase.")

    return data_by_vector_type


#------------------------------------------------------------------------------

import torch
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, roc_auc_score


import pandas as pd
import numpy as np
import os

def guardar_indices_erroneos(X_test, y_test, y_pred, vector_type, model_name, file_names=None):
    """Guarda los índices de vectores mal clasificados en un archivo CSV."""
    # Convertir a arrays de al menos 1 dimensión
    y_test = np.atleast_1d(y_test)
    y_pred = np.atleast_1d(y_pred)

    # Evitar errores por conjuntos vacíos o predicciones sin valores
    if len(y_test) == 0 or len(y_pred) == 0:
        print(f"⚠️ El modelo {model_name} no generó predicciones válidas.")
        return

    # Identificar índices de vectores mal clasificados
    errores_idx = np.where(y_test != y_pred)[0]

    if len(errores_idx) == 0:
        print(f"✅ No se encontraron errores en la clasificación del modelo {model_name} con el vector {vector_type}.")
        return

    # Si hay nombres de archivos, obtener los nombres correspondientes
    if file_names is not None:
        nombres_erroneos = [file_names[i] for i in errores_idx]
    else:
        nombres_erroneos = errores_idx.tolist()  # Si no hay nombres, guardar solo índices

    # Crear DataFrame para guardar
    errores_df = pd.DataFrame({
        'Indice/Nombres': nombres_erroneos,
        'Etiqueta_Real': y_test[errores_idx],
        'Etiqueta_Predicha': y_pred[errores_idx]
    })

    # Guardar en CSV
    errores_dir = './indices_erroneos'
    os.makedirs(errores_dir, exist_ok=True)
    csv_path = os.path.join(errores_dir, f"{model_name}_{vector_type}_indices_erroneos.csv")

    errores_df.to_csv(csv_path, index=False)
    print(f"❗️ Se encontraron {len(errores_idx)} vectores mal clasificados. Guardados en: {csv_path}")


def calcular_metricas_promedio(metrics, prefix=''):
    return {
        f"{prefix}accuracy_mean": np.mean([m.get(f"{prefix}accuracy", 0) for m in metrics]),
        f"{prefix}accuracy_std": np.std([m.get(f"{prefix}accuracy", 0) for m in metrics]),
        f"{prefix}sensitivity_mean": np.mean([m.get(f"{prefix}sensitivity", 0) for m in metrics]),
        f"{prefix}sensitivity_std": np.std([m.get(f"{prefix}sensitivity", 0) for m in metrics]),
        f"{prefix}specificity_mean": np.mean([m.get(f"{prefix}specificity", 0) for m in metrics]),
        f"{prefix}specificity_std": np.std([m.get(f"{prefix}specificity", 0) for m in metrics]),
    }

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_pca_fp_fn(X_test, y_test, y_pred, neighbor_indices=None, title="PCA - KNN (FP vs FN)"):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_test)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Índices para cada categoría
    tp_idx = np.where((y_pred == 1) & (y_test == 1))[0]
    tn_idx = np.where((y_pred == 0) & (y_test == 0))[0]
    fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]

    # Visualización con colores diferenciados
    plt.figure(figsize=(10, 6))
    plt.scatter(X_2d[tp_idx, 0], X_2d[tp_idx, 1], c='green', label='✅ TP (1 → 1)', alpha=0.6)
    plt.scatter(X_2d[tn_idx, 0], X_2d[tn_idx, 1], c='blue', label='✅ TN (0 → 0)', alpha=0.6)
    plt.scatter(X_2d[fp_idx, 0], X_2d[fp_idx, 1], c='orange', label='❌ FP (0 → 1)', alpha=0.7)
    plt.scatter(X_2d[fn_idx, 0], X_2d[fn_idx, 1], c='red', label='❌ FN (1 → 0)', alpha=0.7)

    # Marcar vecinos más cercanos con una "X"
    if neighbor_indices is not None:
        neighbor_indices = np.array(neighbor_indices)
        plt.scatter(X_2d[neighbor_indices, 0], X_2d[neighbor_indices, 1],
                    c='black', marker='x', s=100, label='📎 Vecinos más cercanos')

    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def Entrenar_Modelo(mantener_perfectos, model_name, vector_type, entrenamientos, hiperparametros, 
                    X_train_val, X_test, binary_y_train_val, binary_y_test, 
                    n_splits=5, verbose=False, file_names_test=None, file_names_train=None):

    Elegir_Dispositivo(solo_CPU=False, verborrea=True)
    best_params = None
    best_accuracy = -1
    best_model_metrics = None
    resultados_lista = []

    for params in tqdm(hiperparametros, f"Entrenando {model_name}"):

        torch.cuda.empty_cache()
        gc.collect()

        # Crear modelo según el tipo
        try:
            if model_name == 'MLP':
                device = Elegir_Dispositivo(solo_CPU=False, verborrea=False)
                hidden_layers, neurons_first_layer, epochs, dropout_rate, l2_reg, batch_size, patience = params
                model = MLPModel(input_dim=X_train_val.shape[1], hidden_layers=hidden_layers,
                                 neurons_first_layer=neurons_first_layer, epochs=epochs, lr=0.001, 
                                 dropout_rate=dropout_rate, l2_reg=l2_reg,
                                 device=device, vector_type=vector_type, patience=patience)
                best_model = model
                n_splits=5
            elif model_name == 'CNN1D':
                device = Elegir_Dispositivo(solo_CPU=False, verborrea=False)
                model = CNN1D(input_dim=X_train_val.shape[1], kernel_size=params[0], 
                              num_filters=params[1], dropout_rate=params[2], vector_type=vector_type, epochs = 10)
                best_model = model
                n_splits=5
            elif model_name == 'Transformer':
                model = TransformerClassifier(input_dim=X_train_val.shape[1], num_heads=params[0], 
                                              num_layers=params[1], dim_feedforward=params[2], dropout=params[3],
                                              positional_encoding=params[4], num_dense_neurons=params[5], 
                                              vector_type=vector_type, epochs=params[6])
                best_model = model
                n_splits=5
            elif model_name == 'KNN':
                model = KNN(n_neighbors=params[0], weights=params[1])
                best_model = model
                n_splits=-1

            elif model_name == 'SVM':
                model = SVM(input_dim=X_train_val.shape[1], C=params[1], kernel=params[0])
                best_model = model
                n_splits=-1
            elif model_name == 'RandomForest':
                model = RandomForestOptimized(n_estimators=params[0], max_depth=params[2], split_criterion=params[1], n_jobs=8)
                best_model = model
                n_splits=-1
            else:
                print(f"Modelo '{model_name}' no reconocido")
                continue
        except Exception as e:
            print(f" Error al crear el modelo {model_name} con parámetros {params}: {e}")
            continue

        # Evaluación con K-Fold para obtener métricas de entrenamiento y validación
        metrics = evaluate_model_kfold(model, model_name, X_train_val, binary_y_train_val, n_splits=n_splits, verbose=verbose)

        mean_metrics_train = calcular_metricas_promedio(metrics, prefix='train_')
        mean_metrics_val = calcular_metricas_promedio(metrics, prefix='val_')

        # Evaluar el modelo en el conjunto de prueba
        model.fit(X_train_val, binary_y_train_val)  # Entrenar en todo el conjunto de entrenamiento
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(binary_y_test, y_test_pred, plot_confusion_matrix=False)

        resultados_modelo = {
            "model_name": model_name,
            "vector_type": vector_type,
            "hiperparametros": params,

            # Métricas de entrenamiento
            "train_accuracy_mean": mean_metrics_train.get("train_accuracy_mean", 0),
            "train_accuracy_std": mean_metrics_train.get("train_accuracy_std", 0),

            "train_sensitivity_mean": mean_metrics_train.get("train_sensitivity_mean", 0),
            "train_sensitivity_std": mean_metrics_train.get("train_sensitivity_std", 0),
            "train_specificity_mean": mean_metrics_train.get("train_specificity_mean", 0),
            "train_specificity_std": mean_metrics_train.get("train_specificity_std", 0),

            # Métricas de validación
            "val_accuracy_mean": mean_metrics_val.get("val_accuracy_mean", 0),
            "val_accuracy_std": mean_metrics_val.get("val_accuracy_std", 0),
            
            "val_sensitivity_mean": mean_metrics_val.get("val_sensitivity_mean", 0),
            "val_sensitivity_std": mean_metrics_val.get("val_sensitivity_std", 0),
            "val_specificity_mean": mean_metrics_val.get("val_specificity_mean", 0),
            "val_specificity_std": mean_metrics_val.get("val_specificity_std", 0),

            # Métricas de prueba (test)
            "test_accuracy": test_metrics['Accuracy'],
            "test_sensitivity": test_metrics['Sensitivity (Recall)'],
            "test_specificity": test_metrics['Specificity']
        }
        resultados_lista.append(resultados_modelo)

        if mean_metrics_train['train_accuracy_mean'] > best_accuracy:
            best_model = model
            best_accuracy = mean_metrics_train['train_accuracy_mean']
            best_params = params
            best_model_metrics = mean_metrics_train

            guardar_indices_erroneos(X_test, binary_y_test, y_test_pred, vector_type, model_name)

            guardar_vectores_mal_clasificados(
                y_test=binary_y_test.numpy(),
                y_pred=y_test_pred,
                file_names=file_names_test,
                vector_type=vector_type,
                model_name=model_name
            )
            
            y_test_pred = best_model.predict(X_test)
            calculate_metrics(binary_y_test, y_test_pred, plot_confusion_matrix=True, model_name=f'{model_name}_{vector_type}')
            
            # ⬇️ AQUÍ VIENE EL BLOQUE NUEVO
            if model_name == 'RandomForest':
                #best_model.export_tree_graph(tree_index=0, max_depth=3, show_plot=True, save_path=f"Random_Forest_{vector_type}")
                best_model.plot_forest_structure(max_depth=params[2])
            elif model_name == "KNN":
                # Obtener lista de vectores mal clasificados para visualización
                vectores_erroneos = best_model.obtener_vecinos_mal_clasificados(
                    model=best_model,
                    X_test=X_test,
                    y_test=binary_y_test,
                    file_names_test=file_names_test,
                    file_names_train=file_names_train
                )
                
                # ✅ GRAFICADO POSICIONAMIENTO 2D y 3D CON DESTACADOS
                vectores_test_raw = []
                vectores_destacados = []
                
                for i in range(len(X_test)):
                    vector_np = X_test[i].cpu().numpy() if isinstance(X_test, torch.Tensor) else np.array(X_test[i])
                    label_real = binary_y_test[i].item() if isinstance(binary_y_test, torch.Tensor) else binary_y_test[i]
                    nombre_archivo = file_names_test[i]
                    entry = (vector_np, label_real, nombre_archivo)
                    vectores_test_raw.append(entry)
                
                    if y_test_pred[i] != label_real:
                        vectores_destacados.append(nombre_archivo)
                
                # 🚩 Posicionamiento y visualización 2D
                pos2D, _ = posicionar_vectores(vectores_test_raw)
                graficar_posiciones(pos2D, vectores_test_raw, destacados=vectores_destacados)


                vecinos = [307, 249, 297, 115, 284, 292]

                plot_pca_fp_fn(X_test, binary_y_test, y_test_pred, neighbor_indices=vecinos, title="PCA 2D - Train & Test Errors")
                plot_pca_fp_fn_3d(X_train_val, binary_y_train_val, X_test, binary_y_test, y_test_pred, title="PCA 3D - Train & Test Errors")
                
            elif model_name == "MLP":
                best_model.save_metrics_to_csv(f"./resultados/{model_name}_{vector_type}_metrics.csv")
                best_model.interpretar_resultados_mlp( metrics_path=f"./resultados/{model_name}_{vector_type}_metrics.csv", pesos_path="mlp_weights.pth", show_plot=True)
                


        if model_name in ["MLP", "CNN1D", "Transformer"]:
            ruta_pesos_guardar = f"{model_name}_{vector_type}.pth"
            best_model.save_weights(ruta_pesos_guardar)

        torch.cuda.empty_cache()
        gc.collect()
        
    # 🚨 Guardar el gráfico del mejor modelo solo al final
    if model_name in ["MLP", "CNN1D", "Transformer"]:
        ruta_grafico_guardar = f"{model_name}_{vector_type}_grafico.png"
        best_model.plotter.save_plot(f"{model_name} - Mejor Modelo", vector_type, ruta_grafico_guardar)

    # Guardar Resultados en CSV
    resultados_df = pd.DataFrame(resultados_lista)
    resultados_csv_path = f"./resultados/clasificador_{model_name}_{vector_type}.csv"
    resultados_df.to_csv(resultados_csv_path, index=False)

    # Mostrar el mejor modelo
    print("\n✅ Mejores resultados:")
    if best_params is None:
        print(" No se encontraron mejores parámetros.")
    else:
        print(f"Best Params: {best_params}")
        display(best_model_metrics)

    torch.cuda.empty_cache()
    gc.collect()
    return best_model, best_model_metrics, None
    
#------------------------------------------------------------------------------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D
import numpy as np

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def plot_pca_fp_fn_3d(X_train, y_train, X_test, y_test, y_pred, title="PCA 3D - Train & Test Errors"):
    # Combinar para aplicar PCA conjunta
    X_combined = np.vstack((X_train, X_test))
    pca = PCA(n_components=3)
    X_3d_combined = pca.fit_transform(X_combined)

    # Separar proyecciones
    X_train_3d = X_3d_combined[:len(X_train)]
    X_test_3d = X_3d_combined[len(X_train):]

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Índices de clases en train
    train_class0_idx = np.where(y_train == 0)[0]
    train_class1_idx = np.where(y_train == 1)[0]

    # Índices de errores en test
    fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]

    # Crear figura
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar entrenamiento con distinción de clases
    ax.scatter(X_train_3d[train_class0_idx, 0], X_train_3d[train_class0_idx, 1], X_train_3d[train_class0_idx, 2],
               c='blue', label='Train Class 0', alpha=0.3)
    ax.scatter(X_train_3d[train_class1_idx, 0], X_train_3d[train_class1_idx, 1], X_train_3d[train_class1_idx, 2],
               c='green', label='Train Class 1', alpha=0.3)

    # Dibujar errores del test
    ax.scatter(X_test_3d[fp_idx, 0], X_test_3d[fp_idx, 1], X_test_3d[fp_idx, 2],
               c='orange', label='FP (R:0 | P:1)', alpha=0.8)
    ax.scatter(X_test_3d[fn_idx, 0], X_test_3d[fn_idx, 1], X_test_3d[fn_idx, 2],
               c='red', label='FN (R:1 | P:0)', alpha=0.8)

    # Etiquetas
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.legend()
    plt.tight_layout()
    plt.show()



#------------------------------------------------------------------------------

import torch
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_pca(X_train_val, X_test, y_train_val, y_test, vector_type):
    """Función auxiliar para graficar PCA"""
    pca = PCA(n_components=2)
    X_train_val_2d = pca.fit_transform(X_train_val)
    X_test_2d = pca.transform(X_test) if X_test is not None else None

    binary_y_train_val_np = y_train_val.numpy()
    binary_y_test_np = y_test.numpy() if X_test is not None else None

    plt.figure(figsize=(10, 6))

    # Clase 0 y Clase 1 combinadas
    plt.scatter(
        X_train_val_2d[binary_y_train_val_np == 0, 0],
        X_train_val_2d[binary_y_train_val_np == 0, 1],
        label="NO MELANOMA (Train)",
        alpha=0.6,
        marker='o'
    )
    if X_test is not None:
        plt.scatter(
            X_test_2d[binary_y_test_np == 0, 0],
            X_test_2d[binary_y_test_np == 0, 1],
            label="NO MELANOMA (Test)",
            alpha=0.6,
            marker='x'
        )
        plt.scatter(
            X_test_2d[binary_y_test_np == 1, 0],
            X_test_2d[binary_y_test_np == 1, 1],
            label="MELANOMA (Test)",
            alpha=0.6,
            marker='x'
        )
    
    plt.title(f"PCA - {vector_type} (Clases Combinadas)")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def save_plot(self, title, vector_type, save_path):
    plt.figure(figsize=(12, 6))

    # Gráfico de Precisión
    plt.plot(self.epochs, self.train_acc, label='Train Accuracy')
    plt.plot(self.epochs, self.val_acc, label='Validation Accuracy')

    # Configuración adicional del gráficovector_set
    plt.title(f"{title} - {vector_type}")
    plt.xlabel('Épocas')
    plt.ylabel('Métrica')
    plt.legend()
    plt.grid(True)

    # Guardar el gráfico
    plt.savefig(save_path)
    plt.close()  # Cierra el gráfico para liberar memoria
    print(f"📈 Gráfico guardado en '{save_path}'")

def generate_hyperparameters(param_grid, model_name):
    return list(product(*param_grid[model_name].values()))

 # Función para extraer todos los archivos de una carpeta
def Extraer_Archivos(ruta_carpeta):
    archivos = []
    for archivo in tqdm(Path(ruta_carpeta).glob("**/*"), desc='Cargando imágenes'):
        if archivo.is_file():
            archivos.append(archivo)
    return archivos
        
#------------------------------------------------------------------------------#
# Fuera de Ejecutar_Clasificadores — mantener esta función global
def obtener_vectores_por_nombre(data_by_vector_type, nombres_sin_label):
    vectores_encontrados = []
    nombres_limpios = []

    for nombre_objetivo in nombres_sin_label:
        for vector_type, class_dict in data_by_vector_type.items():
            for label, vector_set in class_dict.items():
                for v, y, file_rot in vector_set:
                    if nombre_objetivo in str(file_rot):
                        vectores_encontrados.append(np.array(v))
                        nombres_limpios.append(str(file_rot))
    return vectores_encontrados, nombres_limpios
#------------------------------------------------------------------------------#

def Ejecutar_Clasificadores(mantener_perfectos=True, show_PCA=True, balancear_datos=True, verbose=False, vector_deseado=None):
    torch.cuda.empty_cache()

    vectors_dir = './vectores'
    labels_path = './diagnosticos_corregidos.csv'

    lista = Extraer_Archivos('../melanoma/vectores')
    pacientes = [str(int(str(ruta.name).split('.')[0].split('B')[-1])) for ruta in lista]

    melanoma, nevus = 0, 0
    df = pd.read_csv(labels_path, sep=";")
    for paciente_a_buscar in pacientes:
        fila = df[df['paciente'] == int(paciente_a_buscar)]
        if not fila.empty:
            diagnostico = int(fila.iloc[0]['diagnostico'])
            if diagnostico == 0:
                nevus += 1
            else:
                melanoma += 1

    print(f"Nevus : {nevus}")
    print(f"Melanoma : {melanoma}")

    menor_clase = min(melanoma, nevus)
    data_by_vector_type = load_data(vectors_dir, labels_path, balance_data=balancear_datos, limite_por_clase=menor_clase)
   
    parametros = []
    param_grid = {
         "KNN": { "n_neighbors": [3], "weights": ["distance"] },
        #"RandomForest":{ "n_estimators": [80], "criterion": ["gini"],"max_depth": [33] } 
        
    }
   
    for model_name in param_grid:
        for vector_type, class_dict in data_by_vector_type.items():
            if vector_deseado is not None and vector_type != vector_deseado:
                continue

            X_train, y_train, X_test, y_test = [], [], [], []
            file_names_train, file_names_test = [], []
            vectores_test_raw = []
            for label, vector_set in class_dict.items():
                if not vector_set:
                    continue
            
                for v, y, file_rot in vector_set:
                    nombre_archivo = f"{file_rot}_label{y}"
            
                    # ➤ Entrenamiento si está rotado (90/180/270)
                    if str(file_rot).split('_')[-1].split('.')[0] in ['90', '180', '270']:
                        X_train.append(v)
                        y_train.append(y)
                        file_names_train.append(nombre_archivo)
                    else:
                        # ➤ Test: cualquier otro (rotación 0 u original)
                        X_test.append(v)
                        y_test.append(y)
                        file_names_test.append(nombre_archivo)
                        vectores_test_raw.append((np.array(v), y, nombre_archivo))  # ✅ Asegura numpy



            # Conversión a tensores
            X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
            X_test = torch.tensor(np.array(X_test), dtype=torch.float32)

            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)

            binary_y_train = (y_train > 0).long()
            binary_y_test = (y_test > 0).long()

            print(f"Procesando modelo: {model_name}, Vector: {vector_type}")
            print(f"   ✅ Test solo con imágenes rotación 0: {len(binary_y_test)} muestras")
            print(f"   🧪 Train con el resto: {len(binary_y_train)} muestras")

            # Guardar nombres de archivos de Train y Test en CSV
            output_dir = './nombres_archivos_train_test'
            os.makedirs(output_dir, exist_ok=True)
            
            train_df = pd.DataFrame({"Archivo_Train": file_names_train})
            test_df = pd.DataFrame({"Archivo_Test": file_names_test})
            
            train_csv_path = os.path.join(output_dir, f"rain_files.csv")
            test_csv_path = os.path.join(output_dir, f"test_files.csv")
            
            train_df.to_csv(train_csv_path, index=False)
            test_df.to_csv(test_csv_path, index=False)
            
            print(f"📁 Archivos de entrenamiento guardados en: {train_csv_path}")
            print(f"📁 Archivos de prueba guardados en: {test_csv_path}")

            if show_PCA:
                plot_pca(X_train, X_test, binary_y_train, binary_y_test, vector_type)
            
            hiperparametros = generate_hyperparameters(param_grid, model_name)
            entrenamientos = len(hiperparametros)
            try:
                best_model, best_model_metrics, _ = Entrenar_Modelo(
                    mantener_perfectos, model_name, vector_type, entrenamientos,
                    hiperparametros, X_train, X_test,
                    binary_y_train, binary_y_test, 5, verbose,
                    file_names_test=file_names_test,  # ✅ pasa los nombres aquí
                    file_names_train=file_names_train
                )

                parametros.append(best_model_metrics)
            except Exception as e:
                print(f"❌ Error entrenando el modelo {model_name}: {e}")
                continue

    display(parametros)
    
#------------------------------------------------------------------------------

def calcular_distancias_absolutas_por_pares_filtrado(
    data_by_vector_type,
    vectores_a_comparar,
    nombres_vectores_a_comparar,
    vector_deseado,
    output_path='./distancias_absolutas'
):
    os.makedirs(output_path, exist_ok=True)

    if vector_deseado not in data_by_vector_type:
        print(f"❌ Vector deseado '{vector_deseado}' no encontrado en los datos.")
        return

    class_dict = data_by_vector_type[vector_deseado]

    for idx, vector_a in enumerate(vectores_a_comparar):
        nombre_vector_a = nombres_vectores_a_comparar[idx]
        vector_a_np = np.array(vector_a)

        # Archivos separados para cada tipo de distancia
        output_file_l1 = os.path.join(output_path, f"distancias_L1_{vector_deseado}_{nombre_vector_a}.csv")
        output_file_l2 = os.path.join(output_path, f"distancias_L2_{vector_deseado}_{nombre_vector_a}.csv")

        with open(output_file_l1, 'w') as f_l1, open(output_file_l2, 'w') as f_l2:
            f_l1.write("Vector_Consulta;Vector_Comparado;Clase;Distancia_Absoluta\n")
            f_l2.write("Vector_Consulta;Vector_Comparado;Clase;Distancia_Euclidea\n")

            for label in [0, 1]:
                vector_set = class_dict[label]
                if not vector_set:
                    continue

                for v, _, file_rot in vector_set:
                    if str(file_rot).endswith('_0.csv'):
                        continue  # 🔴 Ignorar vectores terminados en "_0.csv"

                    v_np = np.array(v)
                    distancia_l1 = np.sum(np.abs(vector_a_np - v_np))
                    distancia_l2 = np.linalg.norm(vector_a_np - v_np)

                    f_l1.write(f"{nombre_vector_a};{file_rot};{label};{distancia_l1:.4f}\n")
                    f_l2.write(f"{nombre_vector_a};{file_rot};{label};{distancia_l2:.4f}\n")

        print(f"📁 Guardado: {output_file_l1}")
        print(f"📁 Guardado: {output_file_l2}")


"""
param_grid = {
    "KNN": { "n_neighbors": [3, 5, 10, 12], "weights": ["uniform", "distance"] },
    "Transformer": {
        "num_heads": [8],
        "num_layers": [3, 6],
        "dim_feedforward": [2048],
        "dropout": [0.1, 0.3],
        "positional_encoding": [False],
        "num_dense_neurons": [100, 150, 250],
        "epochs": [15, 25],
        "batch_size": [16, 32, 64],
        "learning_rate": [5e-4, 1e-4]
    },
    "MLP": {
        "hidden_layers": [4, 6],
        "neurons_first_layer": [128, 256],
        "epochs": [50, 70],
        "dropout_rate": [0.3, 0.5],
        "l2_reg": [1e-4, 1e-3],
        "batch_size": [32, 64],
        "patience": [2]
    },
    "CNN1D": { "kernel_size": [3, 5], "": [2, 4], "dropout_rate": [0.3, 0.5] },
    "SVM": { "kernel": ["poly", "rbf", "sigmoid"], 
        'C': [0.01, 0.1, 1.0, 10.0] }, 
    "RandomForest":{ "n_estimators": [80, 100, 120], "criterion": ["gini", "entropy", "log_loss"],"max_depth": [33, 66, 100] } 
}
"""

#------------------------------------------------------------------------------
# Código completo con carga, posicionamiento, visualización y optimización de memoria

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import minimize

# ----------------------- FUNCIONES DE POSICIONAMIENTO ----------------------- #
# Código completo con carga, posicionamiento, visualización y optimización de memoria

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import minimize

# ----------------------- FUNCIONES DE POSICIONAMIENTO ----------------------- #
import numpy as np

def calcular_distancia(v1, v2):
    return np.sum(np.abs(np.array(v1) - np.array(v2)))


def crear_DS_Distancias(vectores):
    n = len(vectores)
    DS_Distancias = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = calcular_distancia(vectores[i], vectores[j])
            DS_Distancias[i][j] = d
            DS_Distancias[j][i] = d
    return DS_Distancias

def punto_aleatorio_en_circunferencia(centro, radio):
    theta = random.uniform(0, 2 * math.pi)
    x = centro[0] + radio * math.cos(theta)
    y = centro[1] + radio * math.sin(theta)
    return [x, y]

def error_total(punto, centros, radios):
    return sum((math.hypot(punto[0] - cx, punto[1] - cy) - r) ** 2 for (cx, cy), r in zip(centros, radios))

def interseccion_circunferencias(centros, radios):
    x0 = sum(c[0] for c in centros) / len(centros)
    y0 = sum(c[1] for c in centros) / len(centros)
    resultado = minimize(error_total, [x0, y0], args=(centros, radios), method='L-BFGS-B')
    return list(resultado.x) if resultado.success else None
    
# ----------------------- POSICIONAMIENTO Y GRAFICADO ----------------------- #
def posicionar_vectores(vectores_raw):
    if not vectores_raw:
        print("⚠️ Lista vacía de vectores. No se puede posicionar.")
        return [], []

    print(f"\n🚀 Posicionando {len(vectores_raw)} vectores...")

    vectores = [np.array(v[0], dtype=np.float32) for v in vectores_raw]

    DS_Distancias = crear_DS_Distancias(vectores)

    posiciones = [[0.0, 0.0]]
    if len(vectores) > 1:
        posiciones.append(punto_aleatorio_en_circunferencia(posiciones[0], DS_Distancias[0][1]))

    for i in range(2, len(vectores)):
        centros = posiciones[:i]
        radios = [DS_Distancias[i][j] for j in range(i)]
        punto = interseccion_circunferencias(centros, radios)
        posiciones.append(punto if punto else [0, 0])

    return posiciones, vectores_raw

def graficar_posiciones(posiciones, vectores_raw, destacados=None):
    if destacados is None:
        destacados = []

    plt.figure(figsize=(8, 8))

    for i, pos in enumerate(posiciones):
        clase = vectores_raw[i][1]
        if clase == 0:
            if vectores_raw[i][2] in destacados:
                color = 'red'
                marcador = 'x'
            else:
                color = 'blue'
                marcador = 'o'
        else:
            if vectores_raw[i][2] in destacados:
                color = 'blue'
                marcador = 'x'
            else:
                color = 'red'
                marcador = 'o'
        plt.plot(pos[0], pos[1], marker=marcador, color=color)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Clase 0', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Clase 1', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='x', color='black', label='Destacado', markersize=10)
    ]
    plt.legend(handles=legend_elements)
    plt.title("Posicionamiento de vectores en 2D (Clase 0 = Azul, Clase 1 = Rojo)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
