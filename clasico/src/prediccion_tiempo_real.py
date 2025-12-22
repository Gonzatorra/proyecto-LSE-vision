import cv2
import numpy as np
from .utils import *
from .procesar_data import preprocesar_imagen


#Rango de color de piel por defecto (HSV)
LOWER_SKIN_DEFAULT = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN_DEFAULT = np.array([20, 255, 255], dtype=np.uint8)


def predecir(rf_model, le, buffer_size=5, wait_ms=50):
    """
    Realiza la prediccion de gestos en tiempo real utilizando un modelo Random Forest previamente entrenado.
    El proceso captura frames desde la camara, extrae el ROI de la mano mediante preprocesamiento, 
    obtiene las caracteristicas del gesto y suaviza las predicciones usando un buffer dinamico.

    El objetivo es mostrar en pantalla la prediccion mas estable del gesto detectado en cada instante,
    para obtener una salida robusta durante la ejecucion en tiempo real.

    Args:
    --------
        - rf_model (RandomForestClassifier): Modelo de Random Forest previamente entrenado para clasificacion de gestos.
        - le (LabelEncoder): Codificador de etiquetas para traducir entre indices numericos y etiquetas de clases.
        - buffer_size (int, opcional): Numero de predicciones recientes a considerar para suavizar la salida. 
                                       Por defecto es 5.
        - wait_ms (int, opcional): Tiempo de espera en milisegundos entre frames. 
                                   Controla la velocidad de visualizacion. Por defecto es 50 ms.

    Proceso:
    --------
    1. Inicializa la camara y verifica su disponibilidad.
    2. Captura frames en tiempo real desde la camara.
    3. Para cada frame:
        a. Preprocesa la imagen con `preprocesar_imagen()` para extraer el ROI (mano).
        b. Si se detecta la mano, obtiene sus coordenadas mediante `obtener_roi()` y dibuja el rectangulo.
        c. Extrae las caracteristicas del ROI con `extraer_features()`.
        d. Realiza la prediccion del gesto con el modelo `rf_model`.
        e. Traduce la prediccion numerica al nombre de la clase con `LabelEncoder`.
        f. Aplica suavizado temporal utilizando un buffer circular de tamaño `buffer_size` 
           para reducir fluctuaciones en la salida.
        g. Muestra la prediccion mas frecuente (mayoria en buffer) sobre el frame.
        h. Si no se detecta una mano, muestra el mensaje "Gesto no detectado".
    4. Visualiza la ventana de prediccion en tiempo real hasta que el usuario presione 'q'.

    Retorna:
    --------
        - None: La funcion no retorna valores; muestra la prediccion visualmente en una ventana 
          y termina al cerrar la camara o presionar la tecla 'q'.
    """

    #Abrimos la camara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara.")
        return

    #Buffer para guardar los frmaes para suavizar predicciones
    buffer_dynamic = []


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Usamos la funcion unificada para obtener ROI
        roi = preprocesar_imagen(frame)
           
        if roi is not None:
            # Dibujar rectángulo sobre la mano:
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            _, coords_roi = obtener_roi(frame, LOWER_SKIN_DEFAULT, UPPER_SKIN_DEFAULT, tamanyo_resize=(64,64))

            if coords_roi:
                x1, y1, x2, y2 = coords_roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            #Extraer features y predecir letra
            features = extraer_features(roi).reshape(1, -1)
            pred_num = rf_model.predict(features)[0]
            pred_label = le.inverse_transform([pred_num])[0]

            #Buffer para suavizar
            buffer_dynamic.append(pred_label)
            if len(buffer_dynamic) > buffer_size:
                buffer_dynamic.pop(0)

            pred_label_display = max(set(buffer_dynamic), key=buffer_dynamic.count)

            cv2.putText(frame, f"Gesto: {pred_label_display}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        else:
            cv2.putText(frame, "Gesto no detectado", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        cv2.imshow("Predicción en tiempo real", frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def run(rf_model, le, buffer_size=5, wait_ms=50):
    predecir(rf_model, le, buffer_size, wait_ms)