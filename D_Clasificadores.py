import os
import pandas as pd
import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Función para cargar datos y etiquetas
def load_data(vectors_dir='./vectors', labels_path='diagnosticos.csv'):
    # Cargar etiquetas
    labels_df = pd.read_csv(labels_path, sep=';')
    labels_dict = dict(zip(labels_df["paciente"].astype(str), labels_df["diagnostico"]))
    
    print(labels_dict)
    # Cargar datos desde los CSV de cada paciente
    X = []
    y = []
    
    for filename in os.listdir(vectors_dir):
        if filename.endswith(".csv"):
            patient_id = filename.split('.')[0]  # Extrae el ID del paciente del nombre del archivo
            patient_id = patient_id[3:]
            patient_id = int(patient_id)
            patient_id = str(patient_id)
            print(patient_id)
            
            if patient_id in labels_dict:
                vector_df = pd.read_csv(os.path.join(vectors_dir, filename))
                feature_vector = vector_df.values.flatten()  # Aplanar para convertir a un solo vector
                X.append(feature_vector)
                y.append(labels_dict[patient_id])

    # Convertir listas a matrices numpy para usar en modelos de sklearn
    X = np.array(X)
    y = np.array(y)
    return X, y

# Función para evaluar el modelo
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, mejores_params, column):
    # Entrenar el modelo en los datos de entrenamiento
    model.fit(X_train, y_train)
    
    # Predecir en todos los conjuntos
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calcular métricas de precisión, sensibilidad y especificidad
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    
    sens_train = recall_score(y_train, y_train_pred, average='binary')
    sens_val = recall_score(y_val, y_val_pred, average='binary')
    sens_test = recall_score(y_test, y_test_pred, average='binary')
    
    spec_train = precision_score(y_train, y_train_pred, average='binary')  # Usamos precision para calcular especificidad
    spec_val = precision_score(y_val, y_val_pred, average='binary')
    spec_test = precision_score(y_test, y_test_pred, average='binary')
    
    f1_train = f1_score(y_train, y_train_pred, average='binary')
    f1_val = f1_score(y_val, y_val_pred, average='binary')
    f1_test = f1_score(y_test, y_test_pred, average='binary')
    
    # Guardar métricas en diccionario
    resultados_fila = {
        'dataset': 'your_dataset_name',  # Reemplaza esto por el nombre del dataset
        'subset': 'train',  # Podrías agregar más subsets si lo deseas
        'etiqueta': column,
        'algoritmo': model_name,
        'Mejores Hiperparámetros': str(mejores_params),
        'Mean Train Accuracy': accuracy_train,
        'Std Train Accuracy': np.std(y_train_pred),
        'Mean Val Accuracy': accuracy_val,
        'Std Val Accuracy': np.std(y_val_pred),
        'Mean Sensibilidad Train': sens_train,
        'Varianza Sensibilidad Train': np.var(y_train_pred),
        'Mean Especificidad Train': spec_train,
        'Varianza Especificidad Train': np.var(y_train_pred),
        'Mean Sensibilidad Val': sens_val,
        'Varianza Sensibilidad Val': np.var(y_val_pred),
        'Mean Especificidad Val': spec_val,
        'Varianza Especificidad Val': np.var(y_val_pred),
        'Accuracy Test': accuracy_test,
        'Mean Sensibilidad Test': sens_test,
        'Varianza Sensibilidad Test': np.var(y_test_pred),
        'Mean Especificidad Test': spec_test,
        'Varianza Especificidad Test': np.var(y_test_pred)
    }
    
    return resultados_fila

# Funciones de optimización para cada clasificador
def optimize_knn(trial, X_train, y_train):
    n_neighbors = trial.suggest_categorical("n_neighbors", [3, 5, 6, 10])
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()

def optimize_svm(trial, X_train, y_train):
    kernel = trial.suggest_categorical("kernel", ["poly", "rbf", "sigmoid"])
    
    svm = SVC(kernel=kernel, random_state=42)
    scores = cross_val_score(svm, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()

def optimize_mlp(trial, X_train, y_train):
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(128,), (256,)])  # Lista de tuplas
    hidden_layers = trial.suggest_categorical("hidden_layers", [4, 6])
    epochs = trial.suggest_categorical("epochs", [50, 70])

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes * hidden_layers,  # Esto asegura que se repita el tamaño de las capas
                        max_iter=epochs, random_state=42)
    scores = cross_val_score(mlp, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()

def optimize_rf(trial, X_train, y_train):
    n_estimators = trial.suggest_categorical("n_estimators", [80, 100, 120])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_depth = trial.suggest_categorical("max_depth", [33, 66, 100])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()

# Entrenamiento de los clasificadores y almacenamiento de resultados
def Entrenar_Clasificadores():
    # Cargar datos y dividir en conjunto de entrenamiento y prueba
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Lista de clasificadores
    classifiers = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'MLP': MLPClassifier(max_iter=100, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1)
    }
    
    # Guardar los resultados de todos los modelos
    results = []
    
    # Entrenar y evaluar cada clasificador en todos los casos (total de 5: uno para cada columna de los vectores)
    columns = ['maximum', 'minimum', 'mean', 'sum', 'median']
    
    for col in columns:
        X_col = X[:, list(range(len(columns))) == columns.index(col)]
        
        for clf_name, clf in classifiers.items():
            # Ejecutar la optimización para obtener los mejores hiperparámetros
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: optimize_knn(trial, X_train, y_train), n_trials=20)  # Cambiar por el clasificador correspondiente
            
            mejores_params = study.best_params
            
            # Evaluar el clasificador y guardar los resultados
            resultados_fila = evaluate_model(clf, X_train, y_train, X_train, y_train, X_test, y_test, clf_name, mejores_params, col)
            results.append(resultados_fila)
    
    # Crear un DataFrame de los resultados
    results_df = pd.DataFrame(results)
    
    # Guardar los resultados en un archivo CSV
    results_df.to_csv('classification_results.csv', index=False)
    print("Resultados guardados en 'classification_results.csv'")

