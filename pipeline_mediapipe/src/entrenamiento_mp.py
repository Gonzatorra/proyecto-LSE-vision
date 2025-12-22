import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def entrenar_modelo_mediapipe(X, y, save_model="pipeline_mediapipe/modelos_mediapipe/rf_model.pkl"):
    """
    Entrena un modelo de Random Forest para clasificacion de gestos de la mano usando los
    landmarks capturados y guarda el modelo junto con el Labelencoder.

    Esta funcion toma como entrada las caracteristicas (X) y etiquetas (y), codifica las
    etiquetas numericamente, divide los datos en conjuntos de entrenamiento y prueba, 
    entrena un Random Forest con ponderacion de clases balanceada y muestra las metricas
    de evaluacion sobre el conjunto de prueba. Finalmente guarda el modelo y el codificador
    para su posterior uso en prediccion en tiempo real.

    Args:
    --------
        - X (np.array): Array que contiene los landmarks de todas las muestras.
        - y (np.array): Array que contiene las etiquetas correspondientes a cada muestra.
        - save_model (str, opcional): Ruta completa donde se guardara el modelo entrenado. 
          Por defecto es "pipeline_mediapipe/modelos_mediapipe/rf_model.pkl".

    Proceso:
    --------
    1. Crea el directorio donde se almacenara el modelo si no existe.
    2. Codifica las etiquetas con LabelEncoder.
    3. Divide los datos en entrenamiento y prueba (80%-20%) con estratificacion.
    4. Entrena un RandomForestClassifier con 200 estimadores y clases balanceadas.
    5. Evalua el modelo sobre el conjunto de prueba mostrando accuracy y reporte de clasificacion.
    6. Guarda el modelo y el LabelEncoder en la ruta especificada.

    Retorna:
    --------
        - None: La funcion no devuelve valores, pero guarda el modelo y el codificador en disco.
    """

    #Crear directorio y codificador
    os.makedirs("pipeline_mediapipe/modelos_mediapipe", exist_ok=True)
    le = LabelEncoder()

    #Aplicar codificador
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y, random_state=111)
    
    #Crear y entrenar modelo
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=111)
    rf.fit(X_train, y_train)
    
    #Hacer predicciones sobre test set
    y_pred = rf.predict(X_test)

    #Metricas
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    #Guardar modelo y codificador
    joblib.dump(rf, save_model)
    joblib.dump(le, save_model.replace(".pkl","_le.pkl"))
    print("Modelo y codificador guardados.")
