import mlflow

# connect to mlflow
mlflow.set_tracking_uri("http://localhost:5000")

run_id = "ea50b8487a9b413ca67adb5cdaed55ae"

model_uri = f"runs:/{run_id}/model"

mlflow.register_model(
    model_uri=model_uri,
    name="waste_classifier"
)

print("Model registered successfully!")