import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "model/cats_vs_dogs.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (160, 160)

st.title("Cats vs Dogs CNN Classifier")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def prepare_image(img: Image.Image):
    # Convert to RGB in case the uploaded image has an alpha channel or is grayscale
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    return arr

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = prepare_image(image)
    prob_dog = float(model.predict(x, verbose=0)[0][0])  # sigmoid probability

    # Decision threshold
    label = "dog" if prob_dog >= 0.5 else "cat"
    confidence = prob_dog if label == "dog" else (1.0 - prob_dog)

    st.subheader("Prediction")
    st.write(f"**Label:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")
    st.write(f"(Raw dog probability: {prob_dog:.4f})")