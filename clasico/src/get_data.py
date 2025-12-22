import os
import cv2
import numpy as np
import sys
from .utils import obtener_roi

#Rango de color de piel por defecto (HSV)
LOWER_SKIN_DEFAULT = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN_DEFAULT = np.array([20, 255, 255], dtype=np.uint8)



def capturar_data(data_dir, letra, lower_skin = LOWER_SKIN_DEFAULT, upper_skin = UPPER_SKIN_DEFAULT, tamanyo_dataset=200, delay_ms=100):
    """
    Captura imagenes de un gesto de la mano para entrenamiento de un modelo.

    La funcion abre la camara, espera a que el usuario pulse 'n' para iniciar la captura,
    y guarda un numero determinado de frames (tamanyo_dataset) de la mano en un directorio
    especifico para la letra o gesto indicado. Durante la captura, solo se guarda la
    región de interes (ROI) donde se detecta la mano usando detección de piel en HSV.

    Args:
        - data_dir (str): Directorio raiz donde se guardaran los datasets por letra.
        - letra (str): Nombre de la letra o gesto que se esta capturando (se crea un subdirectorio).
        - lower_skin (array, opcional): Determina el limite inferior del rango HSV para detectar la piel.
        - upper_skin(array, opcional): Determina el limite superior del rango HSV para detectar la piel.
        - tamanyo_dataset (int, opcional):Cantidad de frames a tomar por letra.
        - delay_ms (int, opcional): Tiempo de espera entre frames en milisegundos para que no sean imagenes tan similares.

        
    Controles del teclado durante la captura:
        - n : iniciar la captura de frames (solo valido antes de comenzar).
        - q : salir de la funcion, liberando la camara y cerrando ventanas.
        - w : salir del programa por completo usando sys.exit(0).

    Proceso:
    --------
    1. Se abre la camara y se espera a que el usuario pulse 'n'.
    2. Se detecta la mano usando un rango HSV predefinido (lower_skin, upper_skin).
    3. Se aplica una mascara para aislar el color de la piel.
    4. Se limpian ruidos con erosion, dilatacion y suavizado con GaussianBlur.
    5. Se busca el contorno mas grande (la mano) y se define un ROI con margen.
    6. Se dibuja un rectangulo verde sobre la mano y se muestra el contador de frames.
    7. Se guarda el ROI en el directorio correspondiente hasta alcanzar tamanyo_dataset.
    8. Se permite interrumpir la captura en cualquier momento con 'q' o 'w'.
    
    Retorna:
    --------
        - None
    """

    #Abrimos webcam
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("No se ha podido abrir la cámara.")
        return

    #Creamos direcctorio necesario
    directorio = os.path.join(data_dir, letra)
    os.makedirs(directorio, exist_ok=True)

    print(f"Prepárate para capturar la letra '{letra}'. Presiona 'n' para empezar, 'q' para salir de esta captura o 'w' para salir del programa.")

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imshow("Captura", frame)
        key = cv2.waitKey(30) & 0xFF
        #Esperamos hasta que el usuario pulse una tecla
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

    #Captura de frames hasta llegar a lo determinado
    f = 0
    while f < tamanyo_dataset:
        ret, frame = capture.read()
        if not ret:
            break

        #Usamos la funcion para obtener ROI (usada durante todo el proyecto para mantener la consistencia)
        roi, coords = obtener_roi(frame, lower_skin, upper_skin, tamanyo_resize=(128, 128))

        if roi is not None:
            #Dibujar rectangulo sobre la mano
            if coords:
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #Guardar ROI
            img_path = os.path.join(directorio, f"{f}.jpg")
            cv2.imwrite(img_path, roi)
            f += 1

            #Mostrar contador de frames capturados
            cv2.putText(frame, f"{f}/{tamanyo_dataset}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        #Mostrar camara y esperar. Posibilidad de interrumpir cuando sea tambien
        cv2.imshow("Captura", frame)
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




def capturar_por_letra(data_dir, letras):
    for letra in letras:
        result = capturar_data(data_dir, letra)

        if result == "w":  # Usuario quiere volver al menu
            print("Volviendo al menú principal...")
            break



def run(data_dir, letras):
    capturar_por_letra(data_dir, letras)