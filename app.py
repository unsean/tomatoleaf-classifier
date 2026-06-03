import streamlit as st
import numpy as np
import cv2
import joblib
import json
import pandas as pd
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
    voting_model = joblib.load(MODELS_DIR / 'best_model.joblib')
    svm_model = joblib.load(MODELS_DIR / 'svm_combined.joblib')
    codebook = joblib.load(MODELS_DIR / 'orb_codebook.joblib')
    idf_weights = np.load(MODELS_DIR / 'idf_weights.npy')
    le = joblib.load(MODELS_DIR / 'label_encoder.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler_combined.joblib')
    models = {
        'SVM + Combined': svm_model,
        'Voting (soft)': voting_model,
    }
    return models, codebook, idf_weights, le, scaler, config

def validate_leaf(img_bgr, min_plant_ratio=0.06, min_blob_ratio=0.02, min_texture_ratio=0.008):
    """
    Cek apakah gambar kemungkinan daun.
    Syarat ketat: blob tanaman BESAR + tekstur (edge ORB keypoint) DI DALAM blob.
    Dinding hijau = blob gede tapi halus -> ditolak.
    """
    h, w = img_bgr.shape[:2]

    # --- 1. Warna tanaman ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hijau_bawah = np.array([25, 30, 30])
    hijau_atas  = np.array([90, 255, 255])
    mask_tanaman = cv2.inRange(hsv, hijau_bawah, hijau_atas)
    ratio_tanaman = np.count_nonzero(mask_tanaman) / (h * w)

    # --- 2. Blob terbesar ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_tanaman, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)
        largest_blob = areas[np.argmax(areas)]
    else:
        largest_idx = 0
        largest_blob = 0
    ratio_blob = largest_blob / (h * w)

    if largest_idx == 0:
        return False, "Bukan daun terdeteksi. Pastikan gambar adalah daun tomat."

    blob_mask = (labels == largest_idx).astype(np.uint8)

    # --- 3. Edge density DI DALAM blob ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_in_blob = np.count_nonzero(edges & blob_mask)
    texture_ratio = edge_in_blob / max(1, largest_blob)

    # --- 4. ORB keypoints DI DALAM blob ---
    # Daun punya banyak sudut/corner (urat, tepi). Furniture halus = sedikit.
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=15)
    kpts, _ = orb.detectAndCompute(gray, mask=blob_mask)
    kp_count = len(kpts) if kpts else 0
    kp_ratio = kp_count / max(1, largest_blob)

    # --- Decision ---
    color_ok = ratio_tanaman >= min_plant_ratio and ratio_blob >= min_blob_ratio
    # Texture: harus BOTH edge + ORB (wall = edge tipis & ORB sedikit, daun = keduanya tinggi)
    edge_ok = texture_ratio >= min_texture_ratio
    orb_ok = kp_ratio >= 0.001
    texture_ok = edge_ok and orb_ok

    if not color_ok or not texture_ok:
        return False, "Bukan daun terdeteksi. Pastikan gambar adalah daun tomat."
    return True, None

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

def predict(img_bgr, model, codebook, idf_weights, le, scaler, config, use_external_scaler=True):
    img_size = config['img_size']
    gray_norm, gray_clahe = preprocess_image(img_bgr, img_size)
    h = extract_hog_feat(gray_clahe, config)
    o = extract_bovw_tfidf(gray_norm, codebook, idf_weights, config['orb_n'])
    combined = np.concatenate([h, o]).reshape(1, -1)
    if use_external_scaler:
        combined = scaler.transform(combined)
    pred_idx = model.predict(combined)[0]
    probs = model.predict_proba(combined)[0]
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
    models, codebook, idf_weights, le, scaler, config = load_artifacts()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

_fc = pd.read_csv(Path('tomato_output/reports/results/final_comparison.csv'))
MODEL_F1 = dict(zip(_fc['Model'], _fc['F1-Score']))

selected_model_name = st.selectbox("Select model", list(models.keys()))
active_model = models[selected_model_name]
st.success(f"Model loaded: **{selected_model_name}** (F1 = {MODEL_F1.get(selected_model_name, 0):.4f})")


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
        st.success(f"Model: {selected_model_name}")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Leaf Image", use_container_width=True)
    with col2:
        is_leaf, err_msg = validate_leaf(img_bgr)
        if not is_leaf:
            st.subheader("Prediction")
            st.error(err_msg)
            return
        with st.spinner("Classifying..."):
            needs_scaler = selected_model_name == 'Voting (soft)'
            label, probs, classes = predict(img_bgr, active_model, codebook, idf_weights, le, scaler, config, use_external_scaler=needs_scaler)
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
    st.caption(f"Model: {selected_model_name} | Features: HOG ({config['hog_orient']} orient) + ORB-TFIDF ({config['bovw_vocab']} words)")

render_diagnostic_lab()
