import os
import cv2
import numpy as np

from .utils import obtener_roi

#Rango de color de piel por defecto (HSV)
LOWER_SKIN_DEFAULT = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN_DEFAULT = np.array([20, 255, 255], dtype=np.uint8)


def preprocesar_imagen(frame, lower_skin = LOWER_SKIN_DEFAULT, upper_skin = UPPER_SKIN_DEFAULT):
    """
    Preprocesa un frame de imagen capturado para extraer la mano, normalizar tamanyo,
    convertir a escala de grises y equalizar el histograma.

    El objetivo es obtener un ROI consistente para entrenamiento 
    y usarlo en prediccion a tiempo real.

    Args:
    --------
        - frame (np.array): Frame BGR a procesar.
        - lower_skin (array): Determina el limite inferior del rango HSV para detectar la piel.
        - upper_skin (array): Determina el limite superior del rango HSV para detectar la piel.

    Proceso:
    --------
    1. Convierte el frame de BGR a HSV para un rango mas robusto de deteccion de piel.
    2. Aplica una mascara con los límites HSV proporcionados (lower_skin, upper_skin).
    3. Limpia la mascara con erosion, dilatacion y GaussianBlur para reducir ruido. Siguiendo tambien el mismo proceso que en get data.
    4. Busca el contorno más grande (la mano) y descarta contornos pequeños (<1000 px).
    5. Define el ROI con un margen alrededor de la mano.
    6. Redimensiona el ROI al tamaño definido (por defecto 64x64 px).
    7. Convierte el ROI a escala de grises para simplificar los datos y enfocar el modelo en la forma.
    8. Equaliza el histograma de la imagen en gris para mejorar el contraste y la consistencia.
    
    Retorna:
    --------
        - gray_roi (np.array): ROI preprocesado en escala de grises listo para entrenamiento o prediccion.
          Retorna None si no se detecta una mano o el contorno es demasiado pequeño.
    """

    #Obtnemos el ROI
    roi, _ = obtener_roi(frame, lower_skin, upper_skin, (64, 64))
    
    if roi is None or roi.size == 0:
        return None
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #Pasar a gris para reducir complejidad y que la importancia este en la forma y no en el color

    #Ajustar niveles de intensidad de manera uniforme
    gray_roi = cv2.equalizeHist(gray_roi)

    return gray_roi



def preprocesar_dataset(data_dir, output_dir):
    """
    Preprocesa todas las imagenes de un dataset de gestos para extraer la mano,
    normalizar tamanyo, convertir a escala de grises y equalizar el histograma.

    Args:
        - data_dir (str): Directorio raiz donde se guardaran los datasets por letra.
        - output_dir (str): Directorio donde se guardaran las imágenes preprocesadas.
    
    Proceso:
    --------
    Para cada imagen en cada subcarpeta de data_dir:
    1. Se carga la imagen.
    2. Se llama a `preprocesar_imagen()` para extraer la mano.
    3. Si se obtiene un ROI valido, se guarda en output_dir.

    Retorna:
    --------
        None. Las imagenes preprocesadas se guardan en output_dir.
    """

    #Creamos directorio necesario
    os.makedirs(output_dir, exist_ok=True)
    letras = os.listdir(data_dir)

    #Recorremos cada letra de los datos y creamos su directorio concreto
    for letra in letras:
        path_letra = os.path.join(data_dir, letra)
        output_letra = os.path.join(output_dir, letra)
        os.makedirs(output_letra, exist_ok=True)

        for file in os.listdir(path_letra):
            img_path = os.path.join(path_letra, file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            #Preprocesamos la imagen
            roi = preprocesar_imagen(frame)
            if roi is not None:
                cv2.imwrite(os.path.join(output_letra, file), roi)

        print(f"Preprocesadas todas las imagenes de la letra '{letra}'")
    print("Preprocesamiento completado.")



def run(data_dir, output_dir):
    preprocesar_dataset(data_dir, output_dir)