
# main.py
from funciones_practica import load_data, preprocess_data, train_model, evaluate_model, log_model
import argparse
from sklearn.model_selection import train_test_split


def main():
    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación médica.')
    parser.add_argument('--data_path', type=str, default=r"C:\Users\Santiago\Desktop\KeepCoding\Entrega Practicas\Entrega-Practica-Despliegue-Algoritmos\healthcare_dataset.csv",
                       help='Ruta al archivo CSV que contiene los datos.')
    args = parser.parse_args()

    # Cargar los datos
    df = load_data(args.data_path)

    # Preprocesar datos
    X_encoded, y_encoded = preprocess_data(df)

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42
    )

    # Entrenar el modelo
    model = train_model(X_train, y_train)

    # Evaluar el modelo
    metrics = evaluate_model(model, X_test, y_test)

    # Visualizar métricas
    print("Precisión:", metrics["accuracy"])
    print("F1 Score:", metrics["f1_score"])

    # Registrar modelo y métricas en MLflow
    log_model(model, metrics, X_encoded, y_encoded)

if __name__ == "__main__":
    main()
