from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = FastAPI()

# Load trained model
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Recreate model architecture manually
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(9, activation='softmax')
])

# Load trained weights only
model.load_weights("models/realwaste_model.h5")

class_names = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]
@app.get("/")
def home():
    return {"message": "Waste Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    confidence = float(np.max(prediction))
    predicted_class = class_names[np.argmax(prediction)]

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2)
    }