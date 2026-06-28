import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import time

WIB = datetime.timezone(datetime.timedelta(hours=7))
# ── Konfigurasi halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kualitas Udara DKI Jakarta",
    page_icon="🌫️",
    layout="wide",
)

# ── Load artefak model ────────────────────────────────────────────────────────
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

# ── Konstanta ─────────────────────────────────────────────────────────────────
LOKASI_JAKARTA = {
    "Bundaran HI":         "Jakarta Pusat",
    "Senayan":             "Jakarta Pusat",
    "Monas / Gambir":      "Jakarta Pusat",
    "Kemayoran":           "Jakarta Pusat",
    "Senen":               "Jakarta Pusat",
    "Kelapa Gading":       "Jakarta Utara",
    "Pluit / Penjaringan": "Jakarta Utara",
    "Tanjung Priok":       "Jakarta Utara",
    "Sunter":              "Jakarta Utara",
    "Jagakarsa":           "Jakarta Selatan",
    "Kemang":              "Jakarta Selatan",
    "Cilandak":            "Jakarta Selatan",
    "Blok M / Kebayoran":  "Jakarta Selatan",
    "Pasar Minggu":        "Jakarta Selatan",
    "Lubang Buaya":        "Jakarta Timur",
    "Pulogadung":          "Jakarta Timur",
    "Cakung":              "Jakarta Timur",
    "Cipinang":            "Jakarta Timur",
    "Jatinegara":          "Jakarta Timur",
    "Kebon Jeruk":         "Jakarta Barat",
    "Grogol":              "Jakarta Barat",
    "Cengkareng":          "Jakarta Barat",
    "Kalideres":           "Jakarta Barat",
    "Tamansari":           "Jakarta Barat",
}

WILAYAH_ICON = {
    "Jakarta Pusat":   "🏛️",
    "Jakarta Utara":   "⚓",
    "Jakarta Selatan": "🌿",
    "Jakarta Timur":   "🏭",
    "Jakarta Barat":   "🌆",
}

# Setiap wilayah dipetakan ke 1 dari 5 stasiun resmi ISPU DKI
WILAYAH_KE_STASIUN = {
    "Jakarta Pusat":   "DKI1",
    "Jakarta Utara":   "DKI2",
    "Jakarta Selatan": "DKI3",
    "Jakarta Timur":   "DKI4",
    "Jakarta Barat":   "DKI5",
}

STASIUN_NAMA = {
    "DKI1": "Bundaran HI",
    "DKI2": "Kelapa Gading",
    "DKI3": "Jagakarsa",
    "DKI4": "Lubang Buaya",
    "DKI5": "Kebon Jeruk",
}

# Keys internal (ASCII) — urutan ini harus sama dengan urutan fitur saat training
FITUR = ["PM10", "PM25", "SO2", "CO", "O3", "NO2"]

# Label untuk tampilan UI
FITUR_DISPLAY = {
    "PM10": "PM10",  "PM25": "PM2.5",
    "SO2":  "SO₂",   "CO":   "CO",
    "O3":   "O₃",    "NO2":  "NO₂",
}

# Ambang batas aman ISPU (µg/m³)
AMBANG = {
    "PM10": 150,   
    "PM25": 55,     
    "SO2":  180,    
    "CO":   8000,   
    "O3":   235,    
    "NO2":  200,    
}


PROFIL_STASIUN = {
    "DKI1": {"PM10": 55, "PM25": 40, "SO2": 25, "CO": 12, "O3": 24, "NO2": 35},
    "DKI2": {"PM10": 60, "PM25": 45, "SO2": 24, "CO": 14, "O3": 27, "NO2": 38},
    "DKI3": {"PM10": 45, "PM25": 32, "SO2": 18, "CO":  9, "O3": 20, "NO2": 22},
    "DKI4": {"PM10": 65, "PM25": 50, "SO2": 28, "CO": 16, "O3": 28, "NO2": 55},  
    "DKI5": {"PM10": 62, "PM25": 46, "SO2": 26, "CO": 15, "O3": 26, "NO2": 37},
}

# ── Helper ────────────────────────────────────────────────────────────────────
def generate_ispu(stasiun: str) -> dict:
    """
    Simulasi pembacaan sensor ISPU: profil historis ± 15% variasi acak.
    Data baru dibuat setiap kali fungsi ini dipanggil (via Refresh atau ganti lokasi).
    → Untuk produksi: ganti isi fungsi ini dengan request ke API ISPU/AQICN.
    """
    base = PROFIL_STASIUN[stasiun]
    rng  = np.random.default_rng()          # fresh RNG tanpa seed = nilai bervariasi
    return {
        k: max(0, int(v * (1 + rng.uniform(-0.15, 0.15))))
        for k, v in base.items()
    }

def badge(val: float, batas: float) -> str:
    return "✅ Normal" if val <= batas else f"⚠️ Melebihi ({batas})"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Info Model")
    st.metric("Algoritma",            "K-Nearest Neighbors")
    st.metric("k Optimal",            meta["k_terbaik"])
    st.metric("Akurasi Test",         f"{meta['akurasi_test']*100:.2f}%")
    st.metric("Akurasi CV (5-Fold)",  f"{meta['akurasi_cv']*100:.2f}%")
    st.caption(f"Dataset : {meta['dataset']}")
    st.caption(f"Train : {meta['train_size']} | Test : {meta['test_size']} baris")
    st.divider()
    st.caption("📌 Ambang batas aman ISPU:")
    for k in FITUR:
        st.caption(f"  • {FITUR_DISPLAY[k]}: {AMBANG[k]} µg/m³")
    st.divider()
    st.caption(
        "📡 **Sumber data:** Estimasi berbasis profil historis "
        "5 stasiun ISPU DKI Jakarta 2023 (DKI1–DKI5)."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌫️ Klasifikasi Kualitas Udara DKI Jakarta")
st.markdown("Berbasis **K-Nearest Neighbors (k-NN)** · Dataset ISPU DKI Jakarta 2023")
st.divider()

# ── Pilih Lokasi ──────────────────────────────────────────────────────────────
st.subheader("📍 Pilih Lokasi")
pilihan = st.selectbox(
    "Kawasan / kelurahan:",
    ["— Pilih lokasi —"] + sorted(LOKASI_JAKARTA.keys()),
    help="Data ISPU dan hasil analisis tampil otomatis setelah lokasi dipilih",
)

if pilihan == "— Pilih lokasi —":
    st.info("👆 Pilih lokasi untuk melihat data ISPU dan hasil analisis secara otomatis.")
    with st.expander("🗺️ Lihat peta stasiun monitoring ISPU DKI Jakarta"):
        st.markdown("""
| Kode  | Stasiun Referensi | Kawasan yang Dipantau |
|-------|-------------------|-----------------------|
| DKI1  | Bundaran HI       | Jakarta Pusat (Senayan, Gambir, Kemayoran, Senen) |
| DKI2  | Kelapa Gading     | Jakarta Utara (Penjaringan, Tanjung Priok, Sunter) |
| DKI3  | Jagakarsa         | Jakarta Selatan (Kemang, Cilandak, Kebayoran, Pasar Minggu) |
| DKI4  | Lubang Buaya      | Jakarta Timur (Pulogadung, Cakung, Cipinang, Jatinegara) |
| DKI5  | Kebon Jeruk       | Jakarta Barat (Grogol, Cengkareng, Kalideres, Tamansari) |
""")
    st.stop()

# ── Auto-proses setelah lokasi dipilih ───────────────────────────────────────
wilayah      = LOKASI_JAKARTA[pilihan]
stasiun      = WILAYAH_KE_STASIUN[wilayah]
ikon         = WILAYAH_ICON[wilayah]
nama_stasiun = STASIUN_NAMA[stasiun]

# Reset data di session state bila lokasi berubah
if st.session_state.get("_lokasi") != pilihan:
    st.session_state["_lokasi"] = pilihan
    st.session_state["_data"]   = None
    st.session_state["_waktu"]  = None

# Info bar + tombol Refresh
col_info, col_btn = st.columns([5, 1])
with col_info:
    now_str = datetime.datetime.now(WIB).strftime("%d %B %Y, %H:%M")
    st.success(
        f"{ikon} **{pilihan}** · {wilayah} · "
        f"Stasiun: **{nama_stasiun}** ({stasiun}) · 🕐 {now_str} WIB"
    )
with col_btn:
    if st.button("🔄 Refresh", use_container_width=True, help="Simulasikan pembacaan sensor baru"):
        st.session_state["_data"]  = None
        st.session_state["_waktu"] = None

# Ambil / generate data ISPU (hanya dipanggil saat data kosong)
if st.session_state["_data"] is None:
    with st.spinner(f"⏳ Mengambil data ISPU Stasiun {stasiun} – {nama_stasiun}..."):
        time.sleep(0.7)
        st.session_state["_data"]  = generate_ispu(stasiun)
        st.session_state["_waktu"] = datetime.datetime.now(WIB)


d  = st.session_state["_data"]   # dict[str → int], key = FITUR
ft = st.session_state["_waktu"]  # datetime objek waktu fetch

# ── Tampilkan Komponen ISPU ───────────────────────────────────────────────────
st.divider()
st.subheader(f"📊 Data ISPU Terkini · {pilihan}")
st.caption(
    f"📡 Stasiun {stasiun} – {nama_stasiun} · "
    f"{ft.strftime('%d %B %Y, %H:%M')} WIB · "
    f"*(Estimasi berbasis profil historis ISPU 2023)*"
)

cols = st.columns(6)
for col, key in zip(cols, FITUR):
    val   = d[key]
    batas = AMBANG[key]
    with col:
        st.metric(
            label       = f"{FITUR_DISPLAY[key]} (µg/m³)",
            value       = val,
            delta       = f"{val - batas:+d} dari batas {batas}",
            delta_color = "inverse",   # negatif → hijau (aman), positif → merah (melebihi)
        )

# ── Prediksi KNN & Hasil Analisis ─────────────────────────────────────────────
st.divider()
st.subheader(f"🔍 Hasil Analisis Kualitas Udara · {pilihan}")

input_arr    = np.array([[d[k] for k in FITUR]])   # urutan harus sesuai training
input_scaled = scaler.transform(input_arr)
prediksi     = model.predict(input_scaled)[0]
proba        = model.predict_proba(input_scaled)[0]
kelas        = meta["label_classes"]

melebihi_label = [FITUR_DISPLAY[k] for k in FITUR if d[k] > AMBANG[k]]

col_narasi, col_detail = st.columns([1, 1], gap="large")

# ── Kolom kiri: narasi & verdict ──────────────────────────────────────────────
with col_narasi:
    tgl_fmt = ft.strftime("%d %B %Y, %H:%M")

    if prediksi == "Sehat":
        st.success(f"## ✅ {pilihan}: Kualitas Udara SEHAT")
        st.markdown(f"""
Analisis KNN terhadap data ISPU dari Stasiun **{nama_stasiun}** ({stasiun})
menunjukkan bahwa kawasan **{pilihan}** ({wilayah}) memiliki kualitas
udara yang **tergolong Sehat** pada **{tgl_fmt} WIB**.

Kadar polutan berada dalam batas aman ISPU. Aktivitas luar ruangan
dapat dilakukan seperti biasa oleh seluruh kalangan masyarakat.
        """)
    else:
        st.error(f"## ⚠️ {pilihan}: Kualitas Udara TIDAK SEHAT")
        st.markdown(f"""
Analisis KNN terhadap data ISPU dari Stasiun **{nama_stasiun}** ({stasiun})
menunjukkan bahwa kawasan **{pilihan}** ({wilayah}) memiliki kualitas
udara yang **tergolong Tidak Sehat** pada **{tgl_fmt} WIB**.

Tindakan yang disarankan untuk masyarakat sekitar:
- 😷 Gunakan masker N95/KN95 saat berada di luar ruangan
- 🏠 Batasi aktivitas luar, terutama untuk lansia dan anak-anak
- 🪟 Tutup ventilasi untuk mencegah polutan masuk ke dalam ruangan
        """)

    if melebihi_label:
        st.warning(f"🔴 Komponen melebihi ambang batas: **{', '.join(melebihi_label)}**")
    else:
        st.info("🟢 Semua komponen berada di bawah ambang batas aman.")

# ── Kolom kanan: probabilitas + tabel ─────────────────────────────────────────
with col_detail:
    st.markdown("#### 📈 Probabilitas Klasifikasi")
    for label, prob in zip(kelas, proba):
        icon = "✅" if label == "Sehat" else "⚠️"
        st.progress(float(prob), text=f"{icon} {label}: {prob*100:.1f}%")

    st.divider()
    st.markdown("#### 🗂️ Ringkasan Komponen ISPU")
    df = pd.DataFrame({
        "Komponen":      [FITUR_DISPLAY[k] for k in FITUR],
        "Nilai (µg/m³)": [d[k]             for k in FITUR],
        "Batas Aman":    [AMBANG[k]         for k in FITUR],
        "Status":        [badge(d[k], AMBANG[k]) for k in FITUR],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        f"📡 {stasiun} ({nama_stasiun}) · {tgl_fmt} WIB | "
        f"KNN k={meta['k_terbaik']} · Akurasi {meta['akurasi_test']*100:.1f}%"
    )