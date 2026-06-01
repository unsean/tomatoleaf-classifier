import streamlit as st
import numpy as np
import cv2
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from skimage.feature import hog

matplotlib.use("Agg")
MODELS_DIR = Path('tomato_output/models')
CONFIG_PATH = MODELS_DIR / 'model_config.json'

@st.cache_resource
def load_artifacts():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = joblib.load(MODELS_DIR / 'best_model.joblib')
    codebook = joblib.load(MODELS_DIR / 'orb_codebook.joblib')
    idf_weights = np.load(MODELS_DIR / 'idf_weights.npy')
    le = joblib.load(MODELS_DIR / 'label_encoder.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler_combined.joblib')
    return model, codebook, idf_weights, le, scaler, config

def preprocess_image(img_bgr, img_size):
    resized = cv2.resize(img_bgr, tuple(img_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_norm = gray.astype(np.float32) / 255.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray).astype(np.float32) / 255.0
    return gray_norm, gray_clahe

def extract_hog_feat(gray_norm, config):
    return hog(gray_norm, orientations=config['hog_orient'],
               pixels_per_cell=tuple(config['hog_ppc']),
               cells_per_block=tuple(config['hog_cpb']),
               visualize=False, feature_vector=True)

def extract_bovw_tfidf(gray_norm, codebook, idf_weights, orb_n):
    gray_u8 = (gray_norm * 255).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=orb_n)
    _, desc = orb.detectAndCompute(gray_u8, None)
    if desc is None or len(desc) == 0:
        hist = np.zeros(codebook.n_clusters, dtype=np.float32)
    else:
        words = codebook.predict(desc.astype(np.float32))
        hist, _ = np.histogram(words, bins=np.arange(codebook.n_clusters + 1))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
    tfidf = hist * idf_weights
    norm = np.linalg.norm(tfidf)
    return tfidf / norm if norm > 0 else tfidf

def predict(img_bgr, model, codebook, idf_weights, le, scaler, config):
    img_size = config['img_size']
    gray_norm, gray_clahe = preprocess_image(img_bgr, img_size)
    h = extract_hog_feat(gray_clahe, config)
    o = extract_bovw_tfidf(gray_norm, codebook, idf_weights, config['orb_n'])
    combined = np.concatenate([h, o]).reshape(1, -1)
    combined_scaled = scaler.transform(combined)
    pred_idx = model.predict(combined_scaled)[0]
    probs = model.predict_proba(combined_scaled)[0]
    label = le.inverse_transform([pred_idx])[0]
    return label, probs, le.classes_

st.set_page_config(page_title="Tomato Disease Classifier", layout="wide")
bg_color = "#1e1e2f"
card_color = "#27293d"
header_color = "#ffffff"
body_color = "#e2e8f0"
accent = "#00e676"

custom_css = f"""
<style>
    .stApp {{background-color: {bg_color}; color: {body_color};}}
    .card {{background-color: {card_color}; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
    h1, h2, h3, h4, h5, h6 {{color: {header_color};}}
    .stProgress > div > div > div {{background-color: {header_color};}}
    .stTabs {{background-color: {card_color};}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

CLASS_DESCRIPTIONS = {
    "Healthy": "No visible disease. The leaf appears healthy and green.",
    "Early_Blight": "Caused by Alternaria solani. Characterized by dark concentric ring spots on leaves.",
    "Late_Blight": "Caused by Phytophthora infestans. Characterized by dark water-soaked lesions.",
}

CLASS_COLORS = {
    "Healthy": "#00e676",
    "Early_Blight": "#ffab40",
    "Late_Blight": "#ff5252",
}

def draw_donut(pct, color, size=3.5):
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor("#27293d")
    ax.set_facecolor("#27293d")
    ax.pie(
        [1], colors=["#3f4156"], radius=1,
        wedgeprops=dict(width=0.22, edgecolor="#27293d"),
        startangle=90,
    )
    ax.pie(
        [pct, 1 - pct], colors=[color, "#27293d"], radius=1,
        wedgeprops=dict(width=0.22, edgecolor="#27293d"),
        startangle=90, counterclock=False,
    )
    ax.text(0, 0, f"{pct * 100:.1f}%", ha="center", va="center",
            fontsize=28, fontweight="bold", color=color)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    plt.tight_layout(pad=0)
    return fig

try:
    model, codebook, idf_weights, le, scaler, config = load_artifacts()
    st.success(f"Model loaded: **{config['overall_best']}** (F1 = {config['f1']:.4f})")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


def render_diagnostic_lab():
    uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
    img_bgr = None
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.info("Upload an image or select a sample to begin.")
        return

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Image")
        st.success("Model loaded: Voting (soft) (F1 = 0.9058)")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Leaf Image", use_container_width=True)
    with col2:
        with st.spinner("Classifying..."):
            label, probs, classes = predict(img_bgr, model, codebook, idf_weights, le, scaler, config)
        st.subheader("Prediction")
        color = CLASS_COLORS.get(label, "#ffffff")
        st.markdown(
            f"<div style='background-color:{color}22; border-left:4px solid {color}; "
            f"border-radius:6px; padding:0.75rem 1rem; margin:0.5rem 0;'>"
            f"<span style='font-size:1.1rem; font-weight:700; color:{color};'>"
            f"{label.replace('_', ' ')}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"_{CLASS_DESCRIPTIONS.get(label, '')}_")
        DISPLAY_ORDER = ["Healthy", "Early_Blight", "Late_Blight"]
        prob_dict = dict(zip(classes, probs))
        donut_cols = st.columns(len(DISPLAY_ORDER))
        for i, cls in enumerate(DISPLAY_ORDER):
            prob = prob_dict.get(cls, 0)
            clr = CLASS_COLORS.get(cls, "#ffffff")
            fig = draw_donut(prob, clr)
            with donut_cols[i]:
                st.pyplot(fig, use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center; font-size:1.1rem; font-weight:600; "
                    f"color:{clr}; margin-top:-0.5rem;'>{cls.replace('_', ' ')}</p>",
                    unsafe_allow_html=True,
                )
            plt.close(fig)

    st.divider()
    st.caption(f"Model: {config['overall_best']} | Features: HOG ({config['hog_orient']} orient) + ORB‑TFIDF ({config['bovw_vocab']} words)")

render_diagnostic_lab()
