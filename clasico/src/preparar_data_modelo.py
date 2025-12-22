import cv2
import numpy as np
import os
import random
from .utils import extraer_features

FEATURES_FILE = 'features.npz'


def augmentation(img):
    """
    Aplica transformaciones aleatorias de data augmentation a la imagen de la mano
    para aumentar la variabilidad del dataset y asi mejorar la capacidad de generalizacion
    del modelo para evitar overfitting.
    
    Transformaciones aplicadas:
    ---------------------------
    1. Flip horizontal (50% de probabilidad)
       - Invierte la imagen horizontalmente para tener simetria de gestos.

    2. Rotacion aleatoria (+-15 grados)
       - Rota la imagen alrededor del centro, para tener variaciones de angulo.

    3. Cambio de brillo y contraste
       - Ajusta el contraste multiplicando por un factor alpha (0.8 a 1.2).
       - Ajusta el brillo sumando un valor beta (-20 a 20).

    Args:
    -----
        img (array): Imagen RGB de entrada a la que se aplicaran las transformaciones.

    Returns:
    --------
        img (array): Imagen transformadas.
    
    """
    #Obtenemos tamaño de la imagen (filas y columnas)
    rows, cols = img.shape[:2]
    
    if len(img.shape) == 2:  #Grayscale
        img_aug = img.copy()
    else:
        img_aug = img.copy()

    #Flip horizontal con 50% de probabilidad
    if random.random() < 0.5:
        img_aug  = cv2.flip(img_aug, 1)

    #Rotacion aleatoria +-15 grados
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_aug = cv2.warpAffine(img_aug , M, (cols, rows))

    #Cambio de brillo/contraste
    alpha = random.uniform(0.8, 1.2)  #Contraste
    beta = random.randint(-20, 20)    #Brillo
    img_aug = cv2.convertScaleAbs(img_aug , alpha=alpha, beta=beta)

    return img_aug 






def construir_dataset(data_dir, augment=True, augment_factor=5):
    """Construye dataset con posibilidad de aplicar tecnicas de data augmentation.
    Esto sirve para preparar un dataset para entrenar un modelo.
    
    Args:
    -----
        - data_dir (str): Directorio principal que contiene las carpetas por clase
        - augment (bool, opcional): Si es True, aplica data augmentation a cada imagen
        - augment_factor (int, opcional): Número de imágenes sintéticas generadas por cada imagen original

    
    Proceso:
    --------
        1. Recorre cada carpeta (clase) dentro de data_dir
        2. Carga cada imagen
        3. Extrae un vector de caracteristicas con extraer_features() de src
        4. Si augment=True, genera imagenes adicionales con augmentation()
        5. Guarda el dataset completo en un archivo features.npz

    Returns:
    --------
        X (array): Matriz de caracteristicas 
        y (darray): Vector de etiquetas correspondientes a cada muestra
    

    """
    X, y = [], []
    labels = os.listdir(data_dir)

    for label in labels:
        path_label = os.path.join(data_dir, label)
        for file in os.listdir(path_label):
            img_path = os.path.join(path_label, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #Ya estan preprocesadas
            #Si no hay imagen no hacemos nada
            if img is None:
                continue
            
            feats = extraer_features(img)
            X.append(feats)
            y.append(label)


            #Si se quiere aplicar augmentacion
            if augment:
                for _ in range(augment_factor):
                    aug_img = augmentation(img) #Primero aplicamos augmentacion
                    aug_feats = extraer_features(aug_img) #Extraemos las caracteristicas de las imagenes augmentadas
                    X.append(aug_feats)
                    y.append(label)

    #Construimos X e y para el modelo
    X = np.array(X)
    y = np.array(y)
    np.savez(FEATURES_FILE, X=X, y=y)
    print(f"Dataset listo con {len(X)} muestras y guardado en {FEATURES_FILE}")
    return X, y


def run(data_dir, augment=True, augment_factor=5):
    construir_dataset(data_dir, augment, augment_factor)
