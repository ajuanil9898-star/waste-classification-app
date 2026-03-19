import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os
import json
from datetime import datetime
import argparse
import mlflow
import mlflow.keras

# ----------------------------
# Argument Parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# ----------------------------
# MLflow Setup
# ----------------------------
mlflow.set_experiment("Waste_Classification")

# ----------------------------
# Dataset
# ----------------------------
data_dir = "../data/RealWaste"
img_size = (224, 224)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# Model
# ----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# Training + MLflow Logging
# ----------------------------
with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("optimizer", "adam")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Log metrics
    mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])

    # Log model artifact
    mlflow.keras.log_model(model, "model")

# ----------------------------
# Manual Versioned Saving
# ----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"../models/v_{timestamp}"
os.makedirs(model_dir, exist_ok=True)

model.save(f"{model_dir}/model.keras")

metrics = {
    "train_accuracy": history.history["accuracy"][-1],
    "val_accuracy": history.history["val_accuracy"][-1],
    "train_loss": history.history["loss"][-1],
    "val_loss": history.history["val_loss"][-1],
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE
}

with open(f"{model_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(f"{model_dir}/class_names.json", "w") as f:
    json.dump(class_names, f)

print(f"\nModel saved to {model_dir}")