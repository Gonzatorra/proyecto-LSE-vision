from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def entrenar_random_forest(X, y, test_size=0.2, n_estimators=200, random_state=111):
    """Entrena un modelo de Random Forest.

    Args:
    -----
        X (array): Matriz de caracteristicas
        y (array): Vector de etiquetas de cada muestra
        test_size (float): Proporcion de datos reservados para prueba
        n_estimators (int): Numero de arboles en el Random Forest
        random_state (int): Semilla para reproducibilidad

    Returns:
    --------
        rf (RandomForestClassifier): Modelo Random Forest entrenado
        le (LabelEncoder): Codificador de etiquetas ajustado a las clases de y
    """

    #Convertimos etiquetas de texto (A, B, C,...) en numeros (1, 2, 3,...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    #Hacemos train/test data split manteniendo proporcion de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    #Definimos el random forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
    )

    #Entrenamos el modelo
    rf.fit(X_train, y_train)

    #Guardamos
    os.makedirs("modelos_clasico", exist_ok=True)
    with open("modelos_clasico/random_forest_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("modelos_clasico/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    #Evaluamos
    y_pred = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return rf, le




def run(output_dir):
    from .preparar_data_modelo import construir_dataset

    X, y = construir_dataset(output_dir, augment=True, augment_factor=2)
    rf_model, le = entrenar_random_forest(X, y, test_size=0.2, n_estimators=200, random_state=111)

    #Guardar modelo y label encoder
    with open("modelos_clasico/random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("modelos_clasico/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    return rf_model, le