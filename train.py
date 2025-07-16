import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def main():
    """
    This function trains a simple logistic regression model on the Iris dataset.
    This represents our baseline script before introducing MLflow.
    """
    # Load the Iris dataset from scikit-learn
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters. We'll track this with MLflow later.
    C_param = 1.0
    model = LogisticRegression(C=C_param, max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Logistic Regression model (C={C_param})")
    print(f"  Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()