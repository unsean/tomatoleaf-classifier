import streamlit as st
import numpy as np
import cv2
import joblib
import json
from pathlib import Path
from skimage.feature import hog

# --- Config ---
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
    label = le.inverse_transform([pred_idx])[0]
    return label

# --- UI ---
st.set_page_config(page_title="Tomato Disease Classifier", page_icon="", layout="centered")
st.title("Tomato Leaf Disease Classifier")
st.markdown("""
Upload a tomato leaf image to classify it as **Healthy**, **Early Blight**, or **Late Blight**.

This app uses HOG + ORB (TF-IDF) features with an SVM + KNN Voting ensemble.
""")

CLASS_DESCRIPTIONS = {
    'Healthy': 'No visible disease. The leaf appears healthy and green.',
    'Early_Blight': 'Caused by Alternaria solani. Characterized by dark concentric ring spots on leaves.',
    'Late_Blight': 'Caused by Phytophthora infestans. Characterized by dark water-soaked lesions.',
}

CLASS_COLORS = {
    'Healthy': 'green',
    'Early_Blight': 'orange',
    'Late_Blight': 'red',
}

try:
    model, codebook, idf_weights, le, scaler, config = load_artifacts()
    st.success(f"Model loaded: **{config['overall_best']}** (F1 = {config['f1']:.4f})")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a tomato leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the uploaded image.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        with st.spinner("Classifying..."):
            label = predict(img_bgr, model, codebook, idf_weights, le, scaler, config)

        with col2:
            color = CLASS_COLORS.get(label, 'black')
            st.markdown(f"### Prediction: :{color}[{label.replace('_', ' ')}]")
            st.markdown(f"_{CLASS_DESCRIPTIONS.get(label, '')}_")

        st.divider()
        st.caption(f"Model: {config['overall_best']} | Features: HOG ({config['hog_orient']} orient) + ORB-TFIDF ({config['bovw_vocab']} words)")
