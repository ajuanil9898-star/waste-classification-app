import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model(
    "mlartifacts/1/models/m-5141d0f938f149ef85a03b60d128cf22/artifacts/data/model.keras",
    compile=False
)

print("Rebuilding model (fix compatibility)...")

# Recreate model cleanly
new_model = tf.keras.Sequential()

for layer in model.layers:
    new_model.add(layer)

print("Saving clean H5 model...")

new_model.save("models/final_model.h5", save_format="h5")

print("✅ Model converted successfully!")