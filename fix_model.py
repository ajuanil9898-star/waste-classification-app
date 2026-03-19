import tensorflow as tf

print("Loading original model...")

model = tf.keras.models.load_model(
    "mlartifacts/1/models/m-5141d0f938f149ef85a03b60d128cf22/artifacts/data/model.keras",
    compile=False
)

print("Exporting as TensorFlow SavedModel...")

# 🔥 FINAL FIX
model.export("models/final_model")

print("✅ Done!")