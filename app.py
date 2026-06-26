import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime

# ── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kualitas Udara DKI Jakarta",
    page_icon="🌫️",
    layout="wide",
)

# ── Load artefak model ───────────────────────────────────────────────────────
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

# ── Data wilayah DKI Jakarta ─────────────────────────────────────────────────
LOKASI_JAKARTA = {
    # Jakarta Pusat
    "Bundaran HI":        "Jakarta Pusat",
    "Senayan":            "Jakarta Pusat",
    "Monas / Gambir":     "Jakarta Pusat",
    "Kemayoran":          "Jakarta Pusat",
    "Senen":              "Jakarta Pusat",
    # Jakarta Utara
    "Kelapa Gading":      "Jakarta Utara",
    "Pluit / Penjaringan":"Jakarta Utara",
    "Tanjung Priok":      "Jakarta Utara",
    "Sunter":             "Jakarta Utara",
    # Jakarta Selatan
    "Jagakarsa":          "Jakarta Selatan",
    "Kemang":             "Jakarta Selatan",
    "Cilandak":           "Jakarta Selatan",
    "Blok M / Kebayoran": "Jakarta Selatan",
    "Pasar Minggu":       "Jakarta Selatan",
    # Jakarta Timur
    "Lubang Buaya":       "Jakarta Timur",
    "Pulogadung":         "Jakarta Timur",
    "Cakung":             "Jakarta Timur",
    "Cipinang":           "Jakarta Timur",
    "Jatinegara":         "Jakarta Timur",
    # Jakarta Barat
    "Kebon Jeruk":        "Jakarta Barat",
    "Grogol":             "Jakarta Barat",
    "Cengkareng":         "Jakarta Barat",
    "Kalideres":          "Jakarta Barat",
    "Tamansari":          "Jakarta Barat",
}

WILAYAH_ICON = {
    "Jakarta Pusat":  "🏛️",
    "Jakarta Utara":  "⚓",
    "Jakarta Selatan":"🌿",
    "Jakarta Timur":  "🏭",
    "Jakarta Barat":  "🌆",
}

# Ambang batas aman ISPU (untuk indikator status per komponen)
AMBANG = {
    "PM10":  150,
    "PM2.5":  65,
    "SO₂":    75,
    "CO":     30,
    "O₃":    100,
    "NO₂":   200,
}

def badge_status(nilai, batas):
    return "✅ Normal" if nilai <= batas else f"⚠️ Melebihi ({batas})"

# ── Sidebar: info model ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.metric("Algoritma",                  "K-Nearest Neighbors")
    st.metric("Nilai k Optimal",            meta["k_terbaik"])
    st.metric("Akurasi Test Set",           f"{meta['akurasi_test']*100:.2f}%")
    st.metric("Akurasi CV (5-Fold)",        f"{meta['akurasi_cv']*100:.2f}%")
    st.caption(f"Dataset : {meta['dataset']}")
    st.caption(f"Training: {meta['train_size']} baris | Test: {meta['test_size']} baris")
    st.divider()
    st.caption("📌 Ambang batas aman ISPU:")
    for k, v in AMBANG.items():
        st.caption(f"  • {k}: {v} µg/m³")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Sistem Klasifikasi Kualitas Udara DKI Jakarta")
st.markdown(
    "Berbasis **Algoritma K-Nearest Neighbors (k-NN)** · "
    "Dataset ISPU DKI Jakarta 2023"
)
st.divider()

st.subheader("Pilih Lokasi")

col_loc, col_tgl = st.columns([2, 1])
with col_loc:
    pilihan = st.selectbox(
        "Wilayah pengukuran:",
        options=["— Pilih lokasi —"] + sorted(LOKASI_JAKARTA.keys()),
        help="Pilih nama kawasan / kelurahan di DKI Jakarta",
    )
with col_tgl:
    tanggal = st.date_input(
        "Tanggal pengukuran",
        value=datetime.date.today(),
        help="Tanggal saat data ISPU diambil",
    )

# Hentikan render di sini jika lokasi belum dipilih
if pilihan == "— Pilih lokasi —":
    st.info("👆 Silakan pilih lokasi terlebih dahulu untuk melanjutkan.")
    st.stop()

wilayah = LOKASI_JAKARTA[pilihan]
ikon    = WILAYAH_ICON.get(wilayah, "📍")

st.success(
    f"{ikon} Lokasi dipilih: **{pilihan}** · {wilayah} · "
    f"📅 {tanggal.strftime('%d %B %Y')}"
)
st.divider()

st.subheader(f"Input Data ISPU Terbaru · {pilihan}")
st.caption(
    "Masukkan nilai komponen udara dari hasil pengukuran terkini di lokasi ini. "
    "Gunakan angka dari laporan stasiun ISPU atau sensor lokal."
)

pm10 = st.slider("PM10 (µg/m³)",                      min_value=0, max_value=200, value=56,  step=1)
pm25 = st.slider("PM2.5 (µg/m³)",                    min_value=0, max_value=300, value=75,  step=1)
so2  = st.slider("SO₂ – Sulfur Dioksida (µg/m³)",    min_value=0, max_value=100, value=45,  step=1)
co   = st.slider("CO – Karbon Monoksida (µg/m³)",    min_value=0, max_value=60,  value=10,  step=1)
o3   = st.slider("O₃ – Ozon (µg/m³)",               min_value=0, max_value=100, value=26,  step=1)
no2  = st.slider("NO₂ – Nitrogen Dioksida (µg/m³)",  min_value=0, max_value=60,  value=18,  step=1)

st.markdown("")
analisis_btn = st.button(
    f"🔍 Analisis Kualitas Udara {pilihan}",
    type="primary",
    use_container_width=True,
)

if analisis_btn:
    st.divider()
    st.subheader(f"Hasil Analisis Kualitas Udara: {pilihan}")

    # Prediksi
    input_arr    = np.array([[pm10, pm25, so2, co, o3, no2]])
    input_scaled = scaler.transform(input_arr)
    prediksi     = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]
    kelas        = meta["label_classes"]

    # Komponen yang melebihi ambang batas
    komponen_vals = {
        "PM10":  pm10,
        "PM2.5": pm25,
        "SO₂":   so2,
        "CO":    co,
        "O₃":    o3,
        "NO₂":   no2,
    }
    melebihi = [k for k, v in komponen_vals.items() if v > AMBANG.get(k, 9999)]

    col_narasi, col_detail = st.columns([1, 1], gap="large")

    # ── Kolom kiri: narasi & verdict ─────────────────────────────────────────
    with col_narasi:
        tgl_fmt = tanggal.strftime("%d %B %Y")

        if prediksi == "Sehat":
            st.success(f"## ✅ {pilihan}: Kualitas Udara SEHAT")
            st.markdown(
                f"""
Berdasarkan analisis model KNN terhadap data ISPU yang dimasukkan 
({tgl_fmt}), **{pilihan}** ({wilayah}) menunjukkan kualitas udara 
yang **tergolong Sehat**.

Seluruh atau sebagian besar komponen polutan berada dalam batas aman. 
Udara di kawasan **{pilihan}** saat ini tidak membahayakan kesehatan 
masyarakat, dan aktivitas luar ruangan dapat dilakukan seperti biasa.
                """
            )
            st.balloons()

        else:
            st.error(f"## ⚠️ {pilihan}: Kualitas Udara TIDAK SEHAT")
            st.markdown(
                f"""
Berdasarkan analisis model KNN terhadap data ISPU yang dimasukkan 
({tgl_fmt}), **{pilihan}** ({wilayah}) menunjukkan kualitas udara 
yang **tergolong Tidak Sehat**.

Kadar polutan di kawasan **{pilihan}** melebihi ambang batas aman. 
Masyarakat di sekitar wilayah ini diimbau untuk:
- 😷 Menggunakan masker N95/KN95 saat berada di luar ruangan
- 🏠 Membatasi aktivitas luar ruangan, terutama bagi lansia dan anak-anak
- 🪟 Menutup ventilasi rumah untuk mencegah polutan masuk
                """
            )

        # Peringatan komponen bermasalah
        if melebihi:
            st.warning(
                f"🔴 Komponen yang melebihi ambang batas: **{', '.join(melebihi)}**"
            )
        else:
            st.info("🟢 Semua komponen berada di bawah ambang batas aman.")

    # ── Kolom kanan: probabilitas + tabel ringkasan ──────────────────────────
    with col_detail:
        st.markdown("#### 📈 Probabilitas Klasifikasi")
        for label, prob in zip(kelas, proba):
            icon = "✅" if label == "Sehat" else "⚠️"
            st.progress(float(prob), text=f"{icon} {label}: {prob*100:.1f}%")

        st.divider()
        st.markdown(f"#### 🗂️ Ringkasan Data — {pilihan}")

        df = pd.DataFrame({
            "Komponen":       list(komponen_vals.keys()),
            "Nilai (µg/m³)":  list(komponen_vals.values()),
            "Batas Aman":     [AMBANG.get(k, "—") for k in komponen_vals],
            "Status":         [badge_status(v, AMBANG.get(k, 9999))
                               for k, v in komponen_vals.items()],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(
            f"Lokasi: {pilihan} · {wilayah} · {tgl_fmt} | "
            f"Model: KNN (k={meta['k_terbaik']}) · "
            f"Akurasi {meta['akurasi_test']*100:.1f}%"
        )