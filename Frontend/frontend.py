import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻",
    layout="centered"
)

# 🔥 Title + subtitle
st.title("♻ Waste Classification App")
st.markdown("### 🌍 AI-powered Waste Classification System")
st.write("Upload an image to classify waste type")

# ✅ Backend URL
API_URL = "https://waste-classification-app-dak0.onrender.com"

# 🎯 Class icons
icons = {
    "Plastic": "🧴",
    "Metal": "🔩",
    "Glass": "🍾",
    "Paper": "📄",
    "Cardboard": "📦",
    "Vegetation": "🌿",
    "Food Organics": "🍎",
    "Miscellaneous Trash": "🗑️",
    "Textile Trash": "👕"
}

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("🧠 AI is analyzing the waste..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
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

                    # 🎯 Emoji mapping
                    emoji = icons.get(prediction, "♻️")

                    # ✅ Prediction display
                    st.success(f"{emoji} Prediction: {prediction}")

                    # 📊 Progress bar
                    st.progress(min(int(confidence), 100))

                    # 🎯 Confidence styling
                    if confidence > 80:
                        st.success(f"High Confidence: {confidence:.2f}%")
                    elif confidence > 50:
                        st.warning(f"Medium Confidence: {confidence:.2f}%")
                    else:
                        st.error(f"Low Confidence: {confidence:.2f}%")

                else:
                    st.error(f"Prediction failed: {response.text}")

            except Exception as e:
                st.error(f"Error connecting to API: {e}")