import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model

# Connect to MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# IMPORTANT: also set registry URI
mlflow.set_registry_uri("http://127.0.0.1:5001")

# Create / use experiment
mlflow.set_experiment("Waste Classification")

# Load trained model
model = load_model("models/realwaste_model.h5")

# Start MLflow run
with mlflow.start_run():

    print("Logging model to MLflow...")

    mlflow.keras.log_model(
        model,
        artifact_path="model",
        registered_model_name="WasteClassifierModel"
    )

    print("Model successfully registered!")