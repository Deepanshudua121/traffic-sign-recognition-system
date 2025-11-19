import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# Load model only once (caching)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/best_model.h5")
    return model

# -----------------------------
# Class labels for predictions
# -----------------------------
CLASS_LABELS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tonnes",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles >3.5 tonnes prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no-passing zone",
    42: "End no-passing for >3.5 tonnes"
}

# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.title("🚦 Traffic Sign Recognition System")
    st.write("Upload a traffic sign image and the model will predict its meaning.")

    model = load_model()

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Show image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img_resized = img.resize((30, 30))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100

        # Display results
        st.subheader("🔍 Prediction Result:")
        st.write(f"**Class Index:** {class_idx}")
        st.write(f"**Sign Meaning:** {CLASS_LABELS.get(class_idx, 'Unknown Class')}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    st.info("Model Loaded Successfully ✔")


# Run app
if __name__ == "__main__":
    main()
