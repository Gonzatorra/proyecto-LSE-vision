import joblib
import mediapipe as mp
import cv2
import numpy as np
from .extraccion_caracteristicas_mp import extraer_landmarks

mp_hands = mp.solutions.hands


def prediccion_tiempo_real_mediapipe(model_path="pipeline_mediapipe/modelos_mediapipe/rf_model.pkl"):
    """
    Realiza la prediccion de gestos de la mano en tiempo real utilizando un modelo Random Forest
    previamente entrenado con los landmarks capturados.

    Esta funcion abre la camara, extrae los landmarks de la mano, realiza la prediccion
    de la letra y muestra la prediccion en pantalla. Ademas, utiliza
    un buffer para suavizar las predicciones y reducir fluctuaciones en tiempo real.

    Args:
    --------
        - model_path (str, opcional): Ruta completa del modelo entrenado Random Forest a cargar. 
          Por defecto es "pipeline_mediapipe/modelos_mediapipe/rf_model.pkl".

    Proceso:
    --------
    1. Carga el modelo Random Forest y el labelencoder.
    2. Inicializa la camara para captura de video.
    3. Crea un buffer de predicciones para suavizar la salida.
    4. Por cada frame capturado:
        a. Extrae los landmarks de la mano usando MediaPipe y la funcion `extraer_landmarks`.
        b. Si se detecta la mano, realiza la prediccion y actualiza el buffer.
        c. Calcula la prediccion mas frecuente en el buffer y la muestra sobre el frame.
        d. Si no se detecta la mano, muestra el mensaje "Gesto no detectado".
    5. Muestra la ventana de prediccion en tiempo real hasta que el usuario presione 'q'.
    6. Libera la camara y cierra todas las ventanas al finalizar.

    Retorna:
    --------
        - None: La funcion no devuelve valores; solo muestra la prediccion visual en pantalla.
    """

    #Cargar modelo y codificador
    rf = joblib.load(model_path)
    le = joblib.load(model_path.replace(".pkl","_le.pkl"))
    
    #Abrir camara y configuracion inicial
    cap = cv2.VideoCapture(0)
    buffer_preds = []
    buffer_size = 5
    
    #Crear mp hands para poder detectar la mano
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        while True:
            #Captura de cada frame
            ret, frame = cap.read()
            if not ret:
                break

            #Extraccion de landmarks
            landmarks = extraer_landmarks(frame, hands)
            
            #Si se han detectado landmarks
            if landmarks is not None:
                #Predice el gesto
                pred = rf.predict([landmarks])[0]
                #Lo anyade al buffer
                buffer_preds.append(pred)
                #Eliminar la prediccion mas antigua para anyadir la mas actual
                if len(buffer_preds) > buffer_size:
                    buffer_preds.pop(0)

                #Prediccion mas frecuente
                final_pred = np.bincount(buffer_preds).argmax()
                #Obtenemos la letra (conversion de numero a letra)
                letra = le.inverse_transform([final_pred])[0]
                #Se anyade a la pantalla
                cv2.putText(frame, f"Gesto: {letra}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),2)
            
            #Si no se han detectado landmarks
            else:
                cv2.putText(frame, "Gesto no detectado", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),2)
            

            #Abrir la pantalla
            cv2.imshow("Predicción en tiempo real", frame)

            #Salir si se pulsa la letra 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
