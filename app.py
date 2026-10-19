import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src import config

st.set_page_config(page_title="Food classifier", page_icon="🍲", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { max-width: 980px; padding-top: 2rem; }
    .hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #ff7a59 0%, #ff4d8d 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .35rem 0 0; opacity: .92; font-size: .95rem; }
    .class-pill {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 999px;
        background: #2b2f3a;
        color: #e6e6e6;
        font-size: 12.5px;
    }
    .result-card {
        padding: 1.1rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1f8a5a 0%, #15c17a 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .result-card .tag { font-size: 12px; opacity: .85; text-transform: uppercase; letter-spacing: .06em; }
    .result-card .label { font-size: 1.6rem; font-weight: 700; margin: .15rem 0; text-transform: capitalize; }
    .result-card .conf { font-size: 1rem; opacity: .95; }
    .rank-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .rank-num { width: 22px; color: #9aa0a6; font-weight: 600; }
    .rank-label { width: 150px; font-size: 14px; text-transform: capitalize; }
    .bar-track { flex: 1; height: 10px; background: #2b2f3a; border-radius: 6px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #ff7a59, #ff4d8d); }
    .rank-pct { width: 48px; text-align: right; font-size: 13px; color: #c7c7c7; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("checkpoints/food101_vgg16_final.keras")
    # Warm up the graph now so the first real prediction isn't the slow one.
    dummy = np.zeros((1,) + config.IMAGE_SIZE + (3,), dtype=np.float32)
    model.predict(dummy, verbose=0)
    return model


def predict(model, image: Image.Image):
    image = image.convert("RGB").resize(config.IMAGE_SIZE)
    array = np.array(image, dtype=np.float32)
    array = tf.keras.applications.vgg16.preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    return model.predict(array, verbose=0)[0]


st.markdown(
    f"""
    <div class="hero">
        <h1>🍲 Food classifier</h1>
        <p>VGG16 transfer learning, fine-tuned on {config.NUM_CLASSES} Food-101 classes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write(
        "VGG16 conv base (ImageNet-pretrained) with blocks 1-4 frozen and "
        "block 5 fine-tuned, feeding a custom classification head."
    )
    st.markdown("**Classes**")
    st.markdown(
        "".join(f'<span class="class-pill">{n.replace("_", " ")}</span>' for n in config.SELECTED_CLASSES),
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("[View source on GitHub](https://github.com/Charan1845/vgg16-food-classifier)")

model = load_model()
class_names = config.SELECTED_CLASSES

uploaded = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", width="stretch")

    with col2:
        try:
            with st.spinner("Classifying..."):
                preds = predict(model, image)
            top_k = np.argsort(preds)[::-1][:5]
            top1 = top_k[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="tag">Best match</div>
                    <div class="label">{class_names[top1].replace("_", " ")}</div>
                    <div class="conf">{preds[top1]:.1%} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Top 5 predictions**")
            rows = ""
            for rank, idx in enumerate(top_k, start=1):
                pct = preds[idx] * 100
                rows += f"""
                <div class="rank-row">
                    <div class="rank-num">{rank}</div>
                    <div class="rank-label">{class_names[idx].replace("_", " ")}</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
                    <div class="rank-pct">{pct:.1f}%</div>
                </div>
                """
            st.markdown(rows, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload an image to classify it.")
