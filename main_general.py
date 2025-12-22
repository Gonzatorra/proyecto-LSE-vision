import sys
import os

# Importar los dos menus principales
from clasico import menu_clasico
from pipeline_mediapipe import menu_mediapipe


def main():
    while True:
        print("\n#============================#")
        print("#      MENÚ PRINCIPAL        #")
        print("#============================#")
        print("Selecciona el modo de trabajo:\n")
        print("[0] Salir del programa")
        print("[1] Pipeline clásico (segmentación por color y ROI)")
        print("[2] Pipeline con MediaPipe (landmarks en tiempo real)\n")

        opcion = input("Opción: ")

        if opcion == '0':
            print("Saliendo del programa...")
            sys.exit(0)

        #CLASICO
        elif opcion == '1':
            print("\nHas seleccionado el modo CLÁSICO.\n")
            menu_clasico.main()

        #MEDIAPIPE
        elif opcion == '2':
            print("\nHas seleccionado el modo MEDIAPIPE.\n")
            menu_mediapipe.main()

        else:
            print("Opción no válida. Intenta nuevamente.")


if __name__ == "__main__":
    main()
