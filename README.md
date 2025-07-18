# MLFlowTutorial

## MLflow Tracking
Tracking logs everything about the model training session to be reviewed 
and compared later. Each time a training script is run, a "mlrun" in MLflow 
is created. This is analogous to an entry in a lab notebook.

We should see the following things logged:
* _Parameters_: Save the hyperparameters used in the run (eg. `C`)
* _Metrics_: Save the performance results of the model (like `accuracy`)
* _Artifacts_: Save the output file, most importantly, the trained model itself

### Run Experiments
* Run with default parameters
    `python train.py`
* Run with a variation to the C value
    `python train.py --C 0.5`
* Run with a different max_iter value
    `python train.py --max_iter 300`

### Check out the MLflow UI
```bash
mlflow ui
```

## MLflow Projects
An MLproject is a convention for packaging code to make it reproducible.
It specifies the environment and entry points for running the code.
This allows anyone to run the training with a single command.

### Run the MLproject Locally
This command tells MLflow to run the project using your *currently activated* 
local virtual environment (`.venv`), rather than trying to create a new one. This is
a robust and fast way to run projects locally, as it avoids potential issues
with MLflow locating a specific Conda installation on your system.

`mlflow run . --experiment-name "Iris-Project-Runs" --env-manager=local`

## MLflow Model Registry and Serving
The Model Registry is a centralized hub for model versioning, and MLflow makes it easy to deploy these versioned models as live services.

> **Note:** The concept of model "Stages" (`Staging`, `Production`) has been
> deprecated in favor of a more flexible system of aliases and tags. This
> tutorial uses the modern alias-based approach.

### 1. Register a Model Version
Run the training script with the `--register-model-name` flag. This will log the
model and create `Version 1` of it in the registry under the name `iris-classifier`.

```bash
python train.py --C 0.1 --register-model-name "iris-classifier"
```

### 2. Manage the Model using Aliases in the UI
Go to the MLflow UI (`http://localhost:5000`) to manage your model version.
1.  Click the **Models** tab on the left. You will see `iris-classifier`.
2.  Click on the model name, then on the version you wish to manage (e.g., `Version 1`).
3.  Find the **Aliases** section on the model version page.
4.  Click the pencil icon (edit) to add a new alias. Type `production` and hit enter. A model version can have multiple aliases (e.g., `production`, `champion`).

### 3. Load a Model from the Registry (Optional)
The `predict.py` script demonstrates how an application can load a model directly from the registry by its alias for batch predictions, without needing a live server.

```bash
python predict.py
```

### 4. Deploy the Model as a REST API (Serving) 
The final step is to deploy a registered model as a live REST API endpoint.

#### a) Serve the Production Model 
In a terminal, run the following command. It finds the model with the production alias 
and starts a server. This command will continue to run in your terminal.

```bash
mlflow ui
mlflow models serve -m 'models:/iris-classifier@production' -p 1234 --env-manager=local
```

You will see output like `Uvicorn running on http://127.0.0.1:1234`. The server is now ready.

#### b) Query the Deployed Model
**Open a new, separate terminal** and run the `query_model.py` script. 
It will send sample data to the running server and print the predictions it receives back.

## Connecting this to a Production Environment (Databricks)
Everything we've done so far has used a local MLflow tracking server (the mlruns directory). 
In a real-world setting, you would use a centralized, remote tracking server for 
collaboration and governance. 
Databricks provides a managed, best-in-class MLflow server. 
This section shows you how to connect your local code to a remote Databricks workspace.

### 1. Set up Databricks 
1.  Create a free account: Sign up for the Databricks Community Edition. It's free and doesn't require a credit card. 
2.  Generate a token: Once logged in, go to `Settings` > `(User) Developer` > `Access Tokens`. Generate a new token 
    and copy it somewhere safe. You will not be able to see it again. 

### 2. Configure Your Local Machine 
1.  Install the CLI: Make sure your virtual environment is active and run `pip install databricks-cli`. 
2.  Configure connection: Run `databricks configure --token`.
  * **Databricks Host**: Enter your workspace URL (e.g., `https://community.cloud.databricks.com`).
  * **Token**: Paste the token you generated in the previous step.

### 3. Run an Experiment Remotely 
Now, you can run your training script while logging to Databricks. 
You must set the `MLFLOW_TRACKING_URI` environment variable and provide an experiment name.
This tells MLflow to log to Databricks instead of your local mlruns folder. 

A best practice for experiment naming in Databricks is to use a path that includes your 
username, for example: `/Users/your.email@company.com/mlflow-tutorial`.

```bash
# This special 'databricks' URI tells MLflow to use the connection 
# you just configured. 
MLFLOW_TRACKING_URI=databricks python train.py --C 0.88 -- experiment-name /Users/your.email@company.com/mlflow-tutorial --register-model-name "iris-classifier-remote"
```