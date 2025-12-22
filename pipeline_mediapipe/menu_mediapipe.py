from .src.captura_mp import capturar_por_letra_mediapipe
from .src.construccion_dataset_mp import construir_dataset_mediapipe
from .src.entrenamiento_mp import entrenar_modelo_mediapipe
from .src.prediccion_mp import prediccion_tiempo_real_mediapipe

def main():
    DATA_DIR = "pipeline_mediapipe/data_mediapipe"

    letras = ["A","B","C","CH","D","E","F","G","H","I","J","K","L","LL",
              "M","N","Ñ","O","P","Q","R","RR","S","T","U","V","W","X","Y","Z"]

    while True:
        print("\n#--- MENÚ MEDIAPIPE ---#")
        print("[0] Salir")
        print("[1] Capturar dataset")
        print("[2] Construir dataset")
        print("[3] Entrenar modelo")
        print("[4] Predicción en tiempo real\n")

        opcion = input("Selecciona una opción: ")

        if opcion == '0':
            print("Saliendo del programa.")
            break


        elif opcion == '1':
            # Captura de datos
            capturar_por_letra_mediapipe(DATA_DIR, letras)


        elif opcion == '2':
            try:
                # Construccion de dataset
                X, y = construir_dataset_mediapipe(DATA_DIR)
                print(f"Dataset construido con {len(X)} muestras y {len(set(y))} clases.")
            except Exception as e:
                print(f"Error al construir el dataset: {e}")
            
            
        elif opcion == '3':
            try:
                entrenar_modelo_mediapipe(X, y)

            except Exception as e:
                print(f"Error al entrenar el modelo: {e}")
                

        elif opcion == '4':
            try:
                prediccion_tiempo_real_mediapipe()
            except FileNotFoundError:
                print("No se encontró el modelo entrenado. Entrénalo primero.")
        else:
            print("Opción no válida.")
