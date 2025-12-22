import cv2
import mediapipe as mp
import os
import numpy as np
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

from .extraccion_caracteristicas_mp import extraer_landmarks

def capturar_por_letra_mediapipe(data_dir, letras, tamanyo_dataset=200, delay_ms=30):
    """
    Captura de manera secuencial los landmarks de la mano para un conjunto de letras o gestos definidos, 
    utilizando Mediapipe. 
    
    Esta funcion recorre una lista de letras, utilizando la funcion `capturar_estatico_mediapipe` 
    para cada una de ellas. Permite interrumpir la secuencia presionando la tecla 'w', 
    retornando asi al menú principal sin cerrar el programa completo.
    
    Args:
    --------
        - data_dir (str): Directorio raiz donde se almacenarán los datasets individuales por letra.
        - letras (list[str]): Lista de letras o gestos que se desean capturar.
        - tamanyo_dataset (int, opcional): Numero de muestras a capturar por letra. Por defecto, 200.
        - delay_ms (int, opcional): Retardo entre capturas consecutivas en milisegundos. Por defecto, 30.
    
    Proceso:
    --------
    1. Itera sobre la lista de letras a capturar.
    2. Para cada letra, invoca `capturar_estatico_mediapipe()` con los parametros definidos.
    3. Si durante la captura de alguna letra el usuario presiona 'w', se detiene la iteracion
       y se retorna al menu principal sin finalizar el programa.
    
    Retorna:
    --------
        - None: La funcion no devuelve valores, pero puede interrumpir la secuencia de captura 
          si el usuario decide volver al menu principal.
    """
    for letra in letras:
        result = capturar_estatico_mediapipe(data_dir, letra, tamanyo_dataset, delay_ms)
        if result == "w":  # Usuario quiere volver al menu
            print("Volviendo al menú principal...")
            break





def capturar_estatico_mediapipe(data_dir, letra, tamanyo_dataset=200, delay_ms=30):
    """
    Captura landmarks de la mano mediante Mediapipe y guarda las coordenadas
    de cada muestra como archivos .npy en el directorio correspondiente a la letra o gesto indicado.

    Esta funcion permite recopilar un conjunto de muestras de la posicion de la mano
    asociada a un gesto especifico, facilitando la construccion de un dataset
    para el entrenamiento de modelos de reconocimiento de gestos.

    Args:
    --------
        - data_dir (str): Directorio raiz donde se guardaran los datasets por letra.
        - letra (str): Letra o gesto que se desea capturar.
        - tamanyo_dataset (int, opcional): Numero de muestras (frames) a capturar. Por defecto, 200.
        - delay_ms (int, opcional): Tiempo de espera entre capturas consecutivas en milisegundos. Por defecto, 30.

    Controles de teclado:
    --------
        - 'n': Inicia la captura de muestras.
        - 'q': Cancela la captura actual y retorna al menu.
        - 'w': Finaliza completamente el programa.

    Proceso:
    --------
    1. Inicializa la camara y crea el directorio correspondiente a la letra.
    2. Espera a que el usuario presione 'n' para comenzar la captura.
    3. Durante la captura:
        a. Procesa los frames para detectar la mano.
        b. Dibuja los landmarks detectados sobre el frame.
        c. Extrae las coordenadas con `extraer_landmarks()`.
        d. Guarda las coordenadas en formato .npy dentro del directorio de la letra.
        e. Muestra en pantalla el numero de muestras capturadas.
    4. Permite interrumpir la captura con las teclas 'q' o 'w'.
    5. Libera la camara y cierra todas las ventanas de OpenCV al finalizar.

    Retorna:
    --------
        - None: Si la captura finaliza correctamente o se interrumpe con 'q'.
        - "w": Si el usuario presiona 'w' para volver al menu principal sin cerrar el programa.
    """
    # Abrimos la webcam
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("No se ha podido abrir la cámara.")
        return

    # Creamos directorio
    directorio = os.path.join(data_dir, letra)
    os.makedirs(directorio, exist_ok=True)

    print(f"Prepárate para capturar la letra '{letra}'. Presiona 'n' para empezar, 'q' para salir de esta captura o 'w' para salir del programa.")

    # Esperar a que el usuario pulse 'n' para iniciar
    while True:
        ret, frame = capture.read()

        if not ret:
            break

        cv2.imshow("Captura", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('n'):
            break
        elif key == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            return
        
        elif key == ord('w'):
            capture.release()
            cv2.destroyAllWindows()
            return "w"

    # Captura de frames y extraccion de landmarks
    contador = 0
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        while contador < tamanyo_dataset:
            ret, frame = capture.read()
            if not ret:
                break

            # Dibujar landmarks y extraer coordenadas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                #Extraer coordenadas de landmarks de la mano
                coords = extraer_landmarks(frame, hands)
                if coords is not None:
                    #Guardar en el directorio
                    np.save(os.path.join(directorio, f"{contador}.npy"), coords)
                    contador += 1

            #Mostrar contador de frames en pantalla
            cv2.putText(frame, f"{contador}/{tamanyo_dataset}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Captura", frame)

            #Teclas de control
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                capture.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    capture.release()
    cv2.destroyAllWindows()
    print(f"Captura de la letra '{letra}' completada.")
