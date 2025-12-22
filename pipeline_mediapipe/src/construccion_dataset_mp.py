import os
import numpy as np

def construir_dataset_mediapipe(data_dir="data"):
    """
    Construye el dataset a partir de los archivos .npy generados durante la captura,
    cargando los landmarks de cada gesto y asociandolos con su etiqueta correspondiente.

    Esta funcion recorre las carpetas dentro del directorio raiz especificado, asumiendo que
    cada subcarpeta representa una letra o gesto distinto. De cada subcarpeta se cargan todos
    los archivos .npy y se almacenan en dos arrays paralelos: uno con las caracteristicas (X)
    y otro con las etiquetas (y).

    Args:
    --------
        - data_dir (str, opcional): Directorio raiz que contiene las carpetas de cada letra o gesto.
          Por defecto es "data".

    Proceso:
    --------
    1. Inicializa dos listas vacias, X e y.
    2. Recorre todas las subcarpetas dentro de `data_dir`.
    3. Por cada letra:
        a. Verifica que sea un directorio valido.
        b. Carga los archivos .npy correspondientes a esa letra.
        c. Agrega las muestras al array X y las etiquetas al array y.
    4. Convierte las listas a arrays de NumPy para su uso posterior en entrenamiento.

    Retorna:
    --------
        - X (np.array): Array que contiene los landmarks de todas las muestras capturadas.
        - y (np.array): Array que contiene las etiquetas correspondientes a cada muestra.
    """

    #Construccion de arrays
    X, y = [], []

    #Recorrer cada letra de la lista
    for letra in os.listdir(data_dir):
        letra_dir = os.path.join(data_dir, letra)
        if not os.path.isdir(letra_dir):
            continue

        #Si existe el archivo
        for file in os.listdir(letra_dir):
            #Y es un archivo Numpy
            if file.endswith(".npy"):
                #Anyadimos a cada array correspondiente
                X.append(np.load(os.path.join(letra_dir, file)))
                y.append(letra)

    X = np.array(X)
    y = np.array(y)
    return X, y
