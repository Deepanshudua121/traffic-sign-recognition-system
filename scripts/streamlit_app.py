import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
import io

# ---------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model.h5"
META_DIR = ROOT_DIR / "data" / "Meta"


# ---------------------------------------------------------
# Utilities: model & class names
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: Path):
    """Load Keras model once (cached)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


@st.cache_data
def load_class_mapping(num_classes: int):
    """
    Load human-readable class names from CSV in data/Meta if available.
    Falls back to 'Class 0', 'Class 1', ...
    """
    candidate_files = [
        META_DIR / "signnames.csv",
        META_DIR / "SignNames.csv",
        META_DIR / "classes.csv",
        META_DIR / "Meta.csv",
    ]

    for meta_file in candidate_files:
        if meta_file.exists():
            try:
                df = pd.read_csv(meta_file)

                # Try to guess the columns
                class_col, name_col = None, None
                for col in df.columns:
                    lc = col.lower()
                    if "class" in lc or "id" in lc:
                        class_col = col
                    if (
                        "sign" in lc
                        or "name" in lc
                        or "label" in lc
                        or "meaning" in lc
                    ):
                        name_col = col

                if class_col is not None and name_col is not None:
                    mapping = dict(
                        zip(df[class_col].astype(int), df[name_col].astype(str))
                    )
                    return {i: mapping.get(i, f"Class {i}") for i in range(num_classes)}
            except Exception:
                pass

    # Fallback
    return {i: f"Class {i}" for i in range(num_classes)}


# ---------------------------------------------------------
# Preprocessing & prediction
# ---------------------------------------------------------
def preprocess_image(image: Image.Image, input_shape):
    """
    Make the image look EXACTLY like X_train.npy:
    - RGB
    - 30x30 (or whatever the model input is)
    - pixel values in [0, 255], no normalization
    """
    _, h, w, c = input_shape  # should be (None, 30, 30, 3)

    # Force RGB
    image = image.convert("RGB")

    # Resize to the same size as training data
    image = image.resize((w, h))  # e.g. (30, 30)

    # Convert to numpy array
    arr = np.array(image).astype("float32")  # keep 0–255 scale (NO /255.0 here)

    # Add batch dimension -> (1, h, w, c)
    arr = np.expand_dims(arr, axis=0)

    return arr



def predict_single(model, image: Image.Image, class_names: dict):
    """
    Predict for one image. Returns:
    - top_idx, top_prob, probs (np.array), class_names (dict)
    """
    img_array = preprocess_image(image, model.input_shape)

    preds = model.predict(img_array)
    if preds.ndim == 2:
        preds = preds[0]
    elif preds.ndim != 1:
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    probs = tf.nn.softmax(preds).numpy()
    num_classes = probs.shape[0]
    # Ensure mapping has all indices
    class_names = {i: class_names.get(i, f"Class {i}") for i in range(num_classes)}

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    return top_idx, top_prob, probs, class_names


def predict_batch(model, images, class_names: dict):
    """
    Predict for a list of PIL images.
    Returns probs (N, C) and same class_names.
    """
    input_shape = model.input_shape
    arrays = [preprocess_image(img, input_shape)[0] for img in images]  # strip batch dim
    batch = np.stack(arrays, axis=0)
    preds = model.predict(batch)

    if preds.ndim != 2:
        raise ValueError(f"Unexpected batch prediction shape: {preds.shape}")

    probs = tf.nn.softmax(preds, axis=1).numpy()
    num_classes = probs.shape[1]
    class_names = {i: class_names.get(i, f"Class {i}") for i in range(num_classes)}
    return probs, class_names


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def style_page():
    st.set_page_config(
        page_title="Traffic Sign Recognition",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Light custom CSS to make it feel less "default Streamlit"
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #050816;
            color: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background: #0b1020;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        .stMetric {
            background-color: #111827;
            border-radius: 0.75rem;
            padding: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_model_section():
    st.sidebar.title("⚙️ Model Settings")

    model_path_str = st.sidebar.text_input(
        "Model path (relative to repo root)",
        value=str(DEFAULT_MODEL_PATH.relative_to(ROOT_DIR)),
        help="Default: models/best_model.h5",
    )

    model_path = (ROOT_DIR / model_path_str).resolve()

    try:
        model = load_model(model_path)
    except Exception as e:
        st.sidebar.error(
            f"Could not load model from `{model_path}`.\n\nError: {e}"
        )
        st.stop()

    num_classes = model.output_shape[-1]
    class_names = load_class_mapping(num_classes)

    with st.sidebar.expander("Model Info", expanded=False):
        st.write(f"**Model path:** `{model_path}`")
        st.write(f"**Input shape:** `{model.input_shape}`")
        st.write(f"**Output classes:** `{num_classes}`")

    # 👇 This is the sidebar menu
    mode = st.sidebar.radio(
        "Mode",
        ["Single image", "Multiple images (batch)", "Webcam Live"],
        index=0,
    )

    return model, class_names, mode



def show_prediction_block(image, top_idx, top_prob, probs, class_names, title_suffix=""):
    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        # Display image at a reasonable size so it does not appear extremely pixelated
        display_img = image.resize((120, 120), Image.LANCZOS)
        st.image(display_img, caption="Uploaded Image", use_column_width=False)

    with col_info:
        st.subheader(f"🔎 Prediction Result {title_suffix}")

        predicted_name = class_names.get(top_idx, f"Class {top_idx}")
        st.write(f"**Class Index:** {top_idx}")
        st.write(f"**Sign Meaning:** {predicted_name}")
        st.write(f"**Confidence:** {top_prob * 100:.2f}%")

        # Top-5 probabilities chart
        prob_series = pd.Series(
            probs, index=[class_names[i] for i in range(len(probs))]
        )
        top5 = prob_series.sort_values(ascending=False).head(5)
        st.markdown("**Top-5 probabilities:**")
        st.bar_chart(top5)



def page_single_image(model, class_names):
    st.header("🖼 Single Image Prediction")

    st.write(
        "Upload a traffic sign image (preferably similar to the GTSRB dataset – cropped and not too blurry), "
        "and the model will classify it."
    )

    uploaded_file = st.file_uploader(
        "Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with st.spinner("Running prediction..."):
            top_idx, top_prob, probs, class_names = predict_single(
                model, image, class_names
            )
        show_prediction_block(image, top_idx, top_prob, probs, class_names)
    else:
        st.info("⬆️ Upload an image to get a prediction.")


def page_batch_images(model, class_names):
    st.header("📦 Batch Prediction (Multiple Images)")

    st.write(
        "You can upload multiple images at once. The app will run the model on each image "
        "and show a results table with predicted classes and confidences."
    )

    files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if not files:
        st.info("⬆️ Upload 2–10 images to see batch predictions.")
        return

    # Convert files to PIL images
    images = []
    names = []
    for f in files:
        try:
            img = Image.open(io.BytesIO(f.read()))
            images.append(img)
            names.append(f.name)
        except Exception:
            st.error(f"Could not read image file: {f.name}")

    if not images:
        st.error("No valid images to process.")
        return

    with st.spinner("Running batch prediction..."):
        probs, class_names = predict_batch(model, images, class_names)

    top_indices = np.argmax(probs, axis=1)
    top_probs = probs[np.arange(len(probs)), top_indices]

    rows = []
    for fname, idx, prob in zip(names, top_indices, top_probs):
        rows.append(
            {
                "File": fname,
                "Predicted Class ID": int(idx),
                "Predicted Sign": class_names.get(int(idx), f"Class {int(idx)}"),
                "Confidence (%)": round(float(prob) * 100, 2),
            }
        )

    df = pd.DataFrame(rows)

    st.subheader("Results Table")
    st.dataframe(df, use_container_width=True)

    avg_conf = df["Confidence (%)"].mean()
    st.metric("Average confidence across images", f"{avg_conf:.2f} %")
def page_webcam(model, class_names):
    st.header("📷 Live Webcam Traffic Sign Detection")

    st.write("Use your webcam to capture an image and let the model predict the traffic sign.")

    img_file = st.camera_input("Click **Capture** to take a picture")

    if img_file is not None:
        image = Image.open(img_file)

        with st.spinner("Analyzing..."):
            top_idx, top_prob, probs, class_names = predict_single(model, image, class_names)

        st.success("Prediction Complete 🎉")

        display_img = image.resize((120, 120), Image.LANCZOS)
        st.image(display_img, caption="Captured Image", use_column_width=False)

        predicted_name = class_names.get(top_idx, f"Class {top_idx}")
        st.write(f"**Predicted Sign:** {predicted_name}")
        st.write(f"**Class Index:** {top_idx}")


def main():
    style_page()

    st.title("🚦 Traffic Sign Recognition System")
    st.caption(
        "German Traffic Sign Recognition Benchmark (GTSRB) • CNN + Streamlit • "
        "Single, batch & webcam predictions"
    )

    # Get model, class names and selected mode from sidebar
    model, class_names, mode = sidebar_model_section()

    # 👇 Route to the correct page based on sidebar selection
    if mode == "Single image":
        page_single_image(model, class_names)
    elif mode == "Multiple images (batch)":
        page_batch_images(model, class_names)
    elif mode == "Webcam Live":
        page_webcam(model, class_names)

    st.markdown("---")
    with st.expander("ℹ️ Notes on accuracy & wrong predictions"):
        st.write(
            """
            * Model test accuracy on GTSRB is around **98–99%** on clean test images.
            * Very blurred / noisy / far images may produce lower confidence.
            """
        )


if __name__ == "__main__":
    main()
