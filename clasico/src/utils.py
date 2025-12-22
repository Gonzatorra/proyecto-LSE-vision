import cv2
import numpy as np

def obtener_roi(frame, lower_skin, upper_skin, tamanyo_resize):
    #Uso de HSV (Tono, Saturacion, Brillo) por mayor robusted a detectar colores
    #independientemente del brillo o saturacion. Utilizamos mismo proceso que en
    #get_data.py para una mayor consistencia en el entrenamiento.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #----Mascaras----#
    #Aislamos el color que se parezca a la piel
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #Limpiamos mascara 
    mask = cv2.erode(mask, None, iterations=2) #Erosion: Eliminamos ruido eliminando pixeles blancos pequeños
    mask = cv2.dilate(mask, None, iterations=2) #Dilatacion: Agrandamos pixeles blancos restantes
    mask = cv2.GaussianBlur(mask, (7,7), 0) #Suavizamos bordes, hacemos más uniforme la máscara. Ayuda a detectar contornos

    #----Contornos----#
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Teniendo en cuenta que la mano es el objeto mas grande
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < 1000: 
            return None, None #Si el frame tiene menos de 1000px se ignora el frame
        
        #Definir ROI con la posicion de la mano
        x, y, w, h = cv2.boundingRect(max_contour)
        margin = 10  #Margen para no cortar muy cerca de la mano
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(frame.shape[1], x+w+margin), min(frame.shape[0], y+h+margin)
        roi = frame[y1:y2, x1:x2] #Contiene finalmente la region de interes
    else:
        return None, None  #Saltar si no hay mano detectada

    #Redimensionar la imagen
    roi = cv2.resize(roi, tamanyo_resize) 
    return roi, (x1, y1, x2, y2)




def extraer_features(roi):
    """
    Extrae caracteristicas de una imagen de mano para usarlo en el entrenamiento de modelos de 
    reconocimiento de letras. Buscamos intensidad, forma y momentos para generar un vector de
    caracteristicas robusto para un mejor entrenamiento del modelo.

    Caracteristicas extraidas:
    --------------------------
    1. Histograma de intensidades en escala de grises (64 bins, normalizado)
       - Captura la distribución de brillo y contraste de la mano.

    2. Geometria del contorno principal:
       - Area del contorno
       - Perimetro del contorno
       - Relacion de aspecto (ancho/alto) del bounding box
       - Captura tamaño, forma y orientacion aproximada de la mano
    
    3. Momentos de Hu:
       - Invariantes a traslacion, escala y rotacion.
       - Representan la forma general de la mano de manera consistente

    Args:
    -----
        img (array): Imagen RGB de con la mano

    Returns:
    --------
        features (array): Vector 1D que combina todas las caracteristicas calculadas para que
                            pueda ser usado para entrenar modelos de machine learning.
    """

    
    # Si la imagen tiene 3 canales (BGR), conviértela a gris
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        #Conversion a gris para centranos en la forma y no en el color
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    else:
            # Ya esta en gris
            gray = roi.copy()
    #Calculamos histograma, normalizamos y aplanamos
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Máscara binaria para contornos
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Seleccionamos el contorno mas grande y calculamos su area, perimetro y
    #ratio del rectangulo que rodea la mano
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
    else:
        area = perimeter = aspect_ratio = 0

    #Calculamos los momentos de Hu, invariables a traslacion, escala y rotacion para que no dependa
    #de la posicion de la mano o tamaño, es decir, mayor robustez
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()

    #Devolvemos todas las caracteristicas calculadas para el modelo
    features = np.hstack([hist, [area, perimeter, aspect_ratio], hu_moments])
    return features





