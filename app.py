import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1b2a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 2rem; font-weight: 700; color: #e8f4fd; margin: 0 0 0.5rem 0; }
    .hero p { color: #7a9bbf; font-size: 0.95rem; margin: 0; }
    .hero .badge {
        display: inline-block; background: #0e3460; color: #4da6ff;
        font-size: 0.75rem; font-weight: 600; padding: 0.25rem 0.75rem;
        border-radius: 20px; margin-bottom: 1rem; text-transform: uppercase;
    }
    .section-label {
        font-size: 0.7rem; font-weight: 600; color: #4da6ff;
        text-transform: uppercase; letter-spacing: 1.5px; margin: 1.5rem 0 0.75rem 0;
    }
    .result-box { border-radius: 14px; padding: 1.5rem 2rem; margin-top: 1.5rem; text-align: center; }
    .result-positive { background: linear-gradient(135deg, #3d0b0b, #1a0505); border: 1px solid #7f1d1d; }
    .result-negative { background: linear-gradient(135deg, #0b3d1a, #051a0b); border: 1px solid #14532d; }
    .result-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.4rem; }
    .result-value { font-size: 2rem; font-weight: 700; line-height: 1; }
    .result-prob { font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.75; }
    .stButton > button {
        background: linear-gradient(135deg, #1a6fcf, #0e4fa0); color: white;
        border: none; border-radius: 10px; padding: 0.65rem 2rem;
        font-weight: 600; width: 100%; margin-top: 1rem;
    }
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
        color: #9cb8d8 !important; font-size: 0.88rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model_diabetes.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path), True
        except Exception as e:
            st.error(f"Gagal me-load model: {e}")
            return None, False
    return None, False

model, model_loaded = load_model()

# ─── HERO ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">ML · Klasifikasi</div>
    <h1>🩺 Diabetes Predictor</h1>
    <p>Prediksi risiko diabetes berdasarkan data klinis menggunakan model Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.info("⚠️ Model belum ditemukan atau gagal dimuat.")
    st.stop()

# ─── INPUT FORM ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📋 Data Pasien</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
    age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=35)
    hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with col2:
    smoking_history = st.selectbox("Riwayat Merokok", ["non-smoker", "smoker", "past-smoker"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=15.0, value=5.5, step=0.1)
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=500, value=100)

# ─── PREDICT ─────────────────────────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", use_container_width=True):
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": float(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_history,
        "bmi": float(bmi),
        "HbA1c_level": float(hba1c),
        "blood_glucose_level": float(glucose)
    }])

    try:
        prediction = model.predict(input_data)[0]
        prob_text = ""

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            prob_val = proba[1] * 100
            prob_text = f"<div class='result-prob'>Probabilitas diabetes: <strong>{prob_val:.1f}%</strong></div>"

        if prediction == 1:
            st.markdown(f"""
            <div class="result-box result-positive">
                <div class="result-label" style="color:#f87171;">⚠️ Hasil Prediksi</div>
                <div class="result-value" style="color:#f87171;">Berisiko Diabetes</div>
                {prob_text}
            </div>
            """, unsafe_allow_html=True)
            st.warning("Disarankan untuk konsultasi dengan dokter dan lakukan pemeriksaan lebih lanjut.")
        else:
            st.markdown(f"""
            <div class="result-box result-negative">
                <div class="result-label" style="color:#4ade80;">✅ Hasil Prediksi</div>
                <div class="result-value" style="color:#4ade80;">Tidak Berisiko</div>
                {prob_text}
            </div>
            """, unsafe_allow_html=True)
            st.success("Hasil menunjukkan tidak ada indikasi diabetes. Tetap jaga pola hidup sehat!")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

# ─── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:#3d5a78; font-size:0.8rem;'>Dibuat untuk keperluan LKS · Diabetes Prediction</p>", unsafe_allow_html=True)