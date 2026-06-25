import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kualitas udara DKI Jakarta",
    page_icon="🌫️",
    layout="wide"
)

# ── Load model & scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🌫️ Klasifikasi Kualitas Udara DKI Jakarta")
st.markdown("Berbasis **Algoritma K-Nearest Neighbors (k-NN)** · Data ISPU 2023")
st.divider()

# ── Sidebar: info model ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.metric("Algoritma", "K-Nearest Neighbors")
    st.metric("Nilai k Optimal", meta["k_terbaik"])
    st.metric("Akurasi Test Set", f"{meta["akurasi_test"]*100:.2f}%")
    st.metric("Akurasi Cross-Validation (5-Fold)", f"{meta["akurasi_cv"]*100:.2f}%")
    st.caption(f"Dataset: {meta["dataset"]}")
    st.caption(f"Training: {meta["train_size"]} baris | Test: {meta["test_size"]} baris")

# ── Layout utama: 2 kolom ─────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Input Komponen ISPU")
    st.caption("Geser slider sesuai nilai pengukuran dari stasiun ISPU.")

    pm10 = st.slider("PM10 (μg/m³)",      min_value=0,   max_value=200, value=56,  step=1)
    pm25 = st.slider("PM2.5 (μg/m³)",     min_value=0,   max_value=300, value=75,  step=1)
    so2  = st.slider("SO₂ – Sulfur Dioksida (μg/m³)",    min_value=0, max_value=100, value=45, step=1)
    co   = st.slider("CO – Karbon Monoksida (μg/m³)",    min_value=0, max_value=60,  value=10, step=1)
    o3   = st.slider("O₃ – Ozon (μg/m³)",                min_value=0, max_value=100, value=26, step=1)
    no2  = st.slider("NO₂ – Nitrogen Dioksida (μg/m³)",  min_value=0, max_value=60,  value=18, step=1)

    prediksi_btn = st.button("🔍 Prediksi Kualitas Udara", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Hasil Klasifikasi")

    if prediksi_btn:
        input_arr = np.array([[pm10, pm25, so2, co, o3, no2]])
        input_scaled = scaler.transform(input_arr)
        prediksi = model.predict(input_scaled)[0]
        proba    = model.predict_proba(input_scaled)[0]
        kelas    = meta["label_classes"]

        if prediksi == "Sehat":
            st.success(f"### ✅ Kualitas Udara: SEHAT")
            st.markdown("Kondisi udara dalam batas aman untuk aktivitas luar ruangan.")
        else:
            st.error(f"### ⚠️ Kualitas Udara: TIDAK SEHAT")
            st.markdown("Disarankan untuk mengurangi aktivitas luar ruangan dan menggunakan masker.")

        st.divider()
        st.caption("Probabilitas Prediksi:")
        for label, prob in zip(kelas, proba):
            st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

        # Tabel input
        st.divider()
        st.caption("Nilai input yang digunakan:")
        df_input = pd.DataFrame({
            "Komponen"     : ["PM10", "PM2.5", "SO₂", "CO", "O₃", "NO₂"],
            "Nilai (μg/m³)": [pm10, pm25, so2, co, o3, no2]
        })
        st.dataframe(df_input, use_container_width=True, hide_index=True)
    else:
        st.info("Atur nilai komponen ISPU di sebelah kiri, lalu klik tombol **Prediksi**.")
