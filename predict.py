import mlflow
import pandas as pd

def predict_from_registry(model_name, alias):
    """
    Loads a model from the MLflow Model Registry and makes a prediction.
    """
    # Load the model from the registry using the "models:/" URI scheme.
    # This URI tells MLflow to look in the Model Registry for a model
    # with the given name and alias.
    model_uri = f"models:/{model_name}@{alias}"
    print(f"Loading model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except mlflow.exceptions.MlflowException as e:
        print(f"\nCould not load model. Have you registered the model "
              f"and assigned it the '{alias}' alias in the UI? Error: {e}")
        return

    # Prepare sample data for prediction that matches the model's signature.
    sample_data = pd.DataFrame(
        data=[[5.1, 3.5, 1.4, 0.2],  # Expected class 0 (setosa)
              [6.7, 3.1, 4.7, 1.5],  # Expected class 1 (versicolor)
              [6.9, 3.2, 5.7, 2.3]], # Expected class 2 (virginica)
        columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    )

    # Make a prediction
    predictions = model.predict(sample_data)

    print("\nSample Data:")
    print(sample_data)
    print(f"\nPredictions: {predictions}")

if __name__ == "__main__":
    MODEL_NAME = "iris-classifier"
    # Attempt to load the model with the "production" alias.
    predict_from_registry(model_name=MODEL_NAME, alias="production")