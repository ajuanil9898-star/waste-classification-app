from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

# ✅ Load model using SavedModel (NO KERAS)
model = tf.saved_model.load("models/final_model")
infer = model.signatures["serving_default"]

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
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        image = np.array(image)
        image = np.expand_dims(image, axis=0)

        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        output = infer(input_tensor)

        prediction = list(output.values())[0].numpy()

        confidence = float(np.max(prediction))
        predicted_class = class_names[np.argmax(prediction)]

        return {
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}