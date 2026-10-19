import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from src import config

st.set_page_config(page_title="Food classifier", page_icon=":fork_and_knife:")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("checkpoints/food101_vgg16_final.keras")
    class_names = tfds.builder("food101").info.features["label"].names[: config.NUM_CLASSES]
    # Warm up the graph now so the first real prediction isn't the slow one.
    dummy = np.zeros((1,) + config.IMAGE_SIZE + (3,), dtype=np.float32)
    model.predict(dummy, verbose=0)
    return model, class_names


def predict(model, image: Image.Image):
    image = image.convert("RGB").resize(config.IMAGE_SIZE)
    array = np.array(image, dtype=np.float32)
    array = tf.keras.applications.vgg16.preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    return model.predict(array, verbose=0)[0]


st.title("Food classifier")
st.caption(f"VGG16 transfer learning, fine-tuned on {config.NUM_CLASSES} Food-101 classes")
st.caption("Classes: " + ", ".join(n.replace("_", " ") for n in load_model()[1]))

model, class_names = load_model()

uploaded = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", width="stretch")

    with col2:
        st.subheader("Top predictions")
        try:
            with st.spinner("Classifying..."):
                preds = predict(model, image)
            top_k = np.argsort(preds)[::-1][:5]
            for idx in top_k:
                label = class_names[idx].replace("_", " ")
                st.write(f"**{label}** — {preds[idx]:.1%}")
                st.progress(float(preds[idx]))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload an image to classify it.")
