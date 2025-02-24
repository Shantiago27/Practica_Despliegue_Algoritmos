
# functions.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

def load_data(filepath: str) -> pd.DataFrame:
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Realiza el procesamiento de datos incluyendo One-Hot Encoding y Label Encoding."""
    X = df[['Admission Type', 'Medical Condition', 'Medication']]
    y = df['Test Results']

    # Codificación de variables categóricas
    categorical_cols = ['Admission Type', 'Medical Condition', 'Medication']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    X_encoded = preprocessor.fit_transform(X)

    # Codificar variable objetivo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X_encoded, y_encoded

def train_model(X_train, y_train) -> LogisticRegression:
    """Entrena un modelo de Regresión Logística."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test) -> dict:
    """Evalúa el modelo y devuelve un diccionario con las métricas."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }

    return metrics

def log_model(model, metrics, X_encoded, y_encoded) -> None:
    """Registra el modelo y sus métricas en MLflow."""
    # Inicializar el experimento
    mlflow.set_experiment("Medical_Classifier_Experiment")

    try:
        # Iniciar una nueva corrida
        mlflow.start_run()
        
        max_iter_values = 2000
        cv_folds_values = 10
        
        # Registrar parámetros
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", max_iter_values)
        mlflow.log_param("cv_folds", cv_folds_values)

        # Registrar métricas
        mlflow.log_metrics(metrics)

        # Validación Cruzada
        cv_scores = cross_val_score(model, X_encoded, y_encoded, cv=5, scoring='accuracy')
        mlflow.log_metrics({
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std()
        })

        # Registrar el modelo
        mlflow.sklearn.log_model(
            model,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )

    finally:
        # Finalizar la corrida
        mlflow.end_run()
