import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn

def main(args):
    """
    This function trains a simple logistic regression model on the Iris dataset,
    with MLflow tracking.
    """
    # Load the Iris dataset from scikit-learn
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    # Start an Mlflow run. This is a context manager that handles start and end of run.
    with mlflow.start_run():
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Log the hyperparameters
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)

        # Train the model
        model = LogisticRegression(C=args.C, max_iter=args.max_iter)
        model.fit(X_train, y_train)

        # Evaluate the model and log the metric
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Logistic Regression model (C={args.C})")
        print(f"  Accuracy: {accuracy:.4f}")

        # Log the model as an artifact
        # By setting an input_example, MLflow can infer the model's signature.
        # This is a best practice for model validation and deployment.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris-logreg-model",
            input_example=X_train)
        print(f"Model saved in run {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum number of iterations")
    args = parser.parse_args()
    main(args)
