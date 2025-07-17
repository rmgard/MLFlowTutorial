import requests
import json

def query_endpoint(port=1234):
    """Sends a request to the deployed model endpoint and prints the response."""

    url = f'http://127.0.0.1:{port}/invocations'

    # The input data must be in the pandas "split" orientation.
    # This matches the signature of the model saved by MLflow.
    input_data = {
        "dataframe_split": {
            "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            "data": [
                [5.1, 3.5, 1.4, 0.2],  # Expected: setosa (0)
                [6.7, 3.1, 4.7, 1.5]   # Expected: versicolor (1)
            ]
        }
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, data=json.dumps(input_data), headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Successfully received predictions:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to the model server at {url}")
        print("Please ensure the MLflow model server is running in a separate terminal.")
        print(f"You can start it with: mlflow models serve -m 'models:/iris-classifier@production' -p {port}")
        print(f"Error: {e}")

if __name__ == "__main__":
    query_endpoint()