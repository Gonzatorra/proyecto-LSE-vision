import pickle
from .src import get_data, procesar_data, preparar_data_modelo, entrenamiento, prediccion_tiempo_real



def main():
    DATA_DIR = 'data_clasico/'              #Carpeta con imagenes originales
    OUTPUT_DIR = 'data_processed_clasico/'  #Carpeta donde se guardan las imagenes preprocesadas

    letras = [
        "A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "J", "K", "L", "LL",
        "M", "N", "Ñ", "O", "P", "Q", "R", "RR", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    rf_model = None
    le = None

    while True:
        print("\n#---MENÚ CLÁSICO---#")
        print("¿Qué quieres hacer?")
        print("[0] Cerrar el programa")
        print("[1] Capturar dataset")
        print("[2] Preprocesar datos")
        print("[3] Entrenar modelo")
        print("[4] Probar en tiempo real\n")
        
        opcion = input("Selecciona una opción: ")

        if opcion == '0':
            print("Saliendo del programa.")
            break

        elif opcion == '1':
            # Captura de datos
            get_data.run(DATA_DIR, letras)


        elif opcion == '2':
            try:
                procesar_data.run(DATA_DIR, OUTPUT_DIR)
            except Exception as e:
                print(f"Error al procesar los datos: {e}")

        elif opcion == '3':
            try:
                # Construir dataset con augmentacion
                X, y = preparar_data_modelo.construir_dataset(OUTPUT_DIR, augment=True, augment_factor=2)

                # Entrenar Random Forest
                rf_model, le = entrenamiento.run(OUTPUT_DIR)

                # Guardar modelo y LabelEncoder
                with open("modelos_clasico/random_forest_model.pkl", "wb") as f:
                    pickle.dump(rf_model, f)
                with open("modelos_clasico/label_encoder.pkl", "wb") as f:
                    pickle.dump(le, f)
                print("Modelo y codificador guardados en 'modelos_clasico/'")

            except Exception as e:
                print(f"No hay datos suficientes.")


        elif opcion == '4':
            # Cargar modelo si no esta en memoria
            if rf_model is None or le is None:
                try:
                    with open("modelos/random_forest_model.pkl", "rb") as f:
                        rf_model = pickle.load(f)
                    with open("modelos/label_encoder.pkl", "rb") as f:
                        le = pickle.load(f)

                except FileNotFoundError:
                    print("No se encontró el modelo entrenado. Primero entrena el modelo.")
                    continue

            # Prediccion en tiempo real
            prediccion_tiempo_real.run(rf_model, le, buffer_size=5, wait_ms=50)

        else:
            print("Opción no válida.")