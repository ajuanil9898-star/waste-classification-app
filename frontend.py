import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Waste Classifier", page_icon="♻", layout="centered")

st.title("♻ Waste Classification App")
st.write("Upload an image to classify waste type")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
           response = requests.post(
    "http://localhost:8000/predict",
    files={
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    },
)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            confidence = result["confidence"]

            st.success(f"Prediction: {prediction}")
            st.progress(int(confidence))
            st.info(f"Confidence: {confidence}%")
        else:
            st.error("Prediction failed. Check backend.")