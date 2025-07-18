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

    #Set the experiment name for production env. Create experiment if DNE.
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)

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

        # Handle the registered_model_name argument
        retistered_model_name = args.register_model_name
        if retistered_model_name and retistered_model_name.lower() in ['none', '']:
            registered_model_name = None


        # Log the model as an artifact
        # By setting an input_example, MLflow can infer the model's signature,
        # which is a best practice. If a registered_model_name is provided,
        # this will also create a new version of the model in the Model Registry.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris-logreg-model",
            input_example=X_train,
            registered_model_name=registered_model_name
        )
        print(f"Model artifact saved in run {mlflow.active_run().info.run_id}")
        if registered_model_name:
            print(f"Model registered under name: '{registered_model_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum number of iterations")
    parser.add_argument("--experiment-name", type=str, default=None, help="Name the MLflow experiment.")
    parser.add_argument(
        "--register-model-name", type=str, default="None", help="If provided, register model with this name."
    )
    args = parser.parse_args()
    main(args)
