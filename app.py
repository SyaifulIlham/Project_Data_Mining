import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import time
import requests

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

# ── Load CSV historis 2023 ────────────────────────────────────────────────────
# Path lokal — file CSV ada di repo yang sama dengan app.py
CSV_URL = "data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-2023-komponen-data.csv"

@st.cache_data(ttl=3600)
def load_csv_historis(url: str) -> pd.DataFrame:
    """
    Baca CSV lokal dengan format kolom asli ISPU DKI:
      periode_data, tanggal (YYYY-MM-DD), stasiun (misal 'DKI1 BUNDERAN HI'),
      pm_sepuluh, pm_duakomalima, sulfur_dioksida,
      karbon_monoksida, ozon, nitrogen_dioksida, max, parameter_pencemar
    Nilai '-' dan '---' dianggap data kosong (NaN).
    Kolom stasiun diekstrak menjadi kode DKI1–DKI5.
    """
    # Peta nama kolom CSV → key internal
    COL_MAP = {
        "pm_sepuluh":         "PM10",
        "pm_duakomalima":     "PM25",
        "sulfur_dioksida":    "SO2",
        "karbon_monoksida":   "CO",
        "ozon":               "O3",
        "nitrogen_dioksida":  "NO2",
    }
    try:
        df = pd.read_csv(url, na_values=["-", "---", ""])
        df["tanggal"] = pd.to_datetime(df["tanggal"]).dt.date

        # Ekstrak kode stasiun dari kolom 'stasiun'
        # Format: "DKI1 BUNDERAN HI" → ambil token pertama "DKI1"
        df["stasiun_kode"] = (
            df["stasiun"]
            .str.strip()
            .str.split()
            .str[0]
            .str.upper()
        )

        # Rename kolom ke key internal
        df = df.rename(columns=COL_MAP)

        # Pastikan kolom numerik, isi NaN dengan 0
        for col in COL_MAP.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        return df
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat CSV historis: {e}")
        return pd.DataFrame()

df_historis = load_csv_historis(CSV_URL)

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

# Station ID AQICN — diverifikasi dari halaman stasiun masing-masing
# Sumber: aqicn.org/data-platform/api/A<ID>/
STASIUN_AQICN = {
    "DKI1": "@416794",   # Bundaran HI, Menteng (Jakarta Pusat)
    "DKI2": "@416809",   # Kelapa Gading (Jakarta Utara)
    "DKI3": "@416830",   # Jagakarsa (Jakarta Selatan)
    "DKI4": "@416749",   # Lubang Buaya (Jakarta Timur)
    "DKI5": "@416812",   # Kebon Jeruk / Srengseng (Jakarta Barat)
}

FITUR         = ["PM10", "PM25", "SO2", "CO", "O3", "NO2"]
FITUR_DISPLAY = {
    "PM10": "PM10",  "PM25": "PM2.5",
    "SO2":  "SO₂",   "CO":   "CO",
    "O3":   "O₃",    "NO2":  "NO₂",
}
AMBANG = {
    "PM10": 150, "PM25": 65, "SO2": 75, "CO": 30, "O3": 100, "NO2": 200,
}

# Nama parameter AQICN → key internal
AQICN_MAP = {
    "pm10": "PM10",
    "pm25": "PM25",
    "so2":  "SO2",
    "co":   "CO",
    "o3":   "O3",
    "no2":  "NO2",
}

TANGGAL_HARI_INI = datetime.date.today()
TANGGAL_MULAI    = datetime.date(2023, 1, 1)

# ── Helper: ambil data dari CSV ───────────────────────────────────────────────
def ambil_dari_csv(stasiun: str, tanggal: datetime.date) -> dict | None:
    """
    Cari baris yang cocok di CSV historis.
    Cocokkan berdasarkan 'stasiun_kode' (DKI1–DKI5) dan 'tanggal'.
    """
    if df_historis.empty:
        return None
    baris = df_historis[
        (df_historis["stasiun_kode"] == stasiun) &
        (df_historis["tanggal"] == tanggal)
    ]
    if baris.empty:
        return None
    baris = baris.iloc[0]
    return {k: int(baris[k]) for k in FITUR if k in baris}

# ── Helper: ambil data dari AQICN ─────────────────────────────────────────────
def ambil_dari_aqicn(stasiun: str) -> dict | None:
    """
    Panggil API AQICN untuk data real-time.
    API key diambil dari st.secrets["AQICN_TOKEN"].
    """
    try:
        token      = st.secrets["AQICN_TOKEN"]
        station_id = STASIUN_AQICN[stasiun]
        url        = f"https://api.waqi.info/feed/{station_id}/?token={token}"
        resp       = requests.get(url, timeout=8)
        resp.raise_for_status()
        data       = resp.json()

        if data.get("status") != "ok":
            return None

        iaqi   = data["data"].get("iaqi", {})
        result = {}
        for aqicn_key, internal_key in AQICN_MAP.items():
            val = iaqi.get(aqicn_key, {}).get("v")
            result[internal_key] = int(val) if val is not None else 0

        return result
    except Exception as e:
        st.warning(f"⚠️ Gagal mengambil data AQICN: {e}")
        return None

def badge(val: float, batas: float) -> str:
    return "✅ Normal" if val <= batas else f"⚠️ Melebihi ({batas})"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Info Model")
    st.metric("Algoritma",           "K-Nearest Neighbors")
    st.metric("k Optimal",           meta["k_terbaik"])
    st.metric("Akurasi Test",        f"{meta['akurasi_test']*100:.2f}%")
    st.metric("Akurasi CV (5-Fold)", f"{meta['akurasi_cv']*100:.2f}%")
    st.caption(f"Dataset : {meta['dataset']}")
    st.caption(f"Train : {meta['train_size']} | Test : {meta['test_size']} baris")
    st.divider()

    st.markdown("#### 📡 Sumber Data")
    st.info(
        "**2023** → CSV historis GitHub\n\n"
        "**2024 – hari ini** → API real-time AQICN"
    )
    st.divider()

    st.caption("📌 Ambang batas aman ISPU:")
    for k in FITUR:
        st.caption(f"  • {FITUR_DISPLAY[k]}: {AMBANG[k]} µg/m³")
    st.divider()

    with st.expander("🗺️ Stasiun ISPU DKI"):
        st.markdown("""
| Kode | Stasiun | Wilayah |
|------|---------|---------|
| DKI1 | Bundaran HI | Jakarta Pusat |
| DKI2 | Kelapa Gading | Jakarta Utara |
| DKI3 | Jagakarsa | Jakarta Selatan |
| DKI4 | Lubang Buaya | Jakarta Timur |
| DKI5 | Kebon Jeruk | Jakarta Barat |
""")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌫️ Klasifikasi Kualitas Udara DKI Jakarta")
st.markdown("Berbasis **K-Nearest Neighbors (k-NN)** · Dataset ISPU DKI Jakarta 2023 + Real-time AQICN")
st.divider()

# ── Input: Lokasi & Tanggal ───────────────────────────────────────────────────
st.subheader("📍 Pilih Lokasi & Tanggal")

col_lok, col_tgl = st.columns([2, 1])
with col_lok:
    pilihan = st.selectbox(
        "Kawasan / kelurahan:",
        ["— Pilih lokasi —"] + sorted(LOKASI_JAKARTA.keys()),
        help="Data ISPU dan hasil analisis tampil otomatis setelah lokasi dipilih",
    )
with col_tgl:
    tanggal_input = st.date_input(
        "Tanggal:",
        value=TANGGAL_HARI_INI,
        min_value=TANGGAL_MULAI,
        max_value=TANGGAL_HARI_INI,
        format="DD/MM/YYYY",
        help="2023 → data CSV historis | 2024–hari ini → API real-time AQICN",
    )

if pilihan == "— Pilih lokasi —":
    st.info("👆 Pilih lokasi untuk melihat data ISPU dan hasil analisis secara otomatis.")
    st.stop()

# ── Tentukan sumber data ───────────────────────────────────────────────────────
wilayah      = LOKASI_JAKARTA[pilihan]
stasiun      = WILAYAH_KE_STASIUN[wilayah]
ikon         = WILAYAH_ICON[wilayah]
nama_stasiun = STASIUN_NAMA[stasiun]

tahun_input  = tanggal_input.year
is_realtime  = tahun_input >= 2024
is_today     = tanggal_input == TANGGAL_HARI_INI

if is_realtime:
    sumber_label = "🟢 Real-time · API AQICN"
    sumber_desc  = "Data diambil langsung dari sensor AQICN"
else:
    sumber_label = "📂 Historis · CSV 2023"
    sumber_desc  = "Data diambil dari dataset ISPU DKI Jakarta 2023"

# ── Reset session state bila lokasi / tanggal berubah ─────────────────────────
cache_key = f"{pilihan}_{tanggal_input}"
if st.session_state.get("_cache_key") != cache_key:
    st.session_state["_cache_key"] = cache_key
    st.session_state["_data"]      = None
    st.session_state["_waktu"]     = None
    st.session_state["_sumber"]    = None

# ── Info bar + tombol Refresh ─────────────────────────────────────────────────
col_info, col_btn = st.columns([5, 1])
with col_info:
    tgl_fmt = tanggal_input.strftime("%d %B %Y")
    st.success(
        f"{ikon} **{pilihan}** · {wilayah} · "
        f"Stasiun: **{nama_stasiun}** ({stasiun}) · 📅 {tgl_fmt} · {sumber_label}"
    )
with col_btn:
    if st.button("🔄 Refresh", use_container_width=True, help="Ambil ulang data dari sumber"):
        st.session_state["_data"]   = None
        st.session_state["_waktu"]  = None
        st.session_state["_sumber"] = None

# ── Ambil data (hanya jika belum ada di session) ───────────────────────────────
if st.session_state["_data"] is None:
    if is_realtime:
        # ── Real-time: cek dulu apakah AQICN_TOKEN ada ────────────────────────
        if "AQICN_TOKEN" not in st.secrets:
            st.error(
                "🔑 **API key AQICN belum dikonfigurasi.**\n\n"
                "Tambahkan token Anda di **Streamlit Secrets** dengan format:\n"
                "```toml\nAQICN_TOKEN = \"YOUR_TOKEN_HERE\"\n```\n"
                "Daftar gratis di https://aqicn.org/api/"
            )
            st.stop()

        with st.spinner(f"⏳ Mengambil data real-time AQICN untuk {nama_stasiun}..."):
            data = ambil_dari_aqicn(stasiun)

        if data is None:
            st.error(
                "❌ Gagal mengambil data dari AQICN. "
                "Periksa koneksi internet atau validitas API key Anda."
            )
            st.stop()

        st.session_state["_data"]   = data
        st.session_state["_waktu"]  = datetime.datetime.now()
        st.session_state["_sumber"] = "aqicn"

    else:
        # ── Historis: cari di CSV ──────────────────────────────────────────────
        with st.spinner(f"⏳ Mencari data CSV untuk {stasiun} · {tanggal_input}..."):
            time.sleep(0.4)
            data = ambil_dari_csv(stasiun, tanggal_input)

        if data is None:
            st.warning(
                f"📭 Data untuk **{nama_stasiun} ({stasiun})** "
                f"pada **{tgl_fmt}** tidak ditemukan di CSV.\n\n"
                "Kemungkinan penyebab: tanggal libur / data kosong di dataset."
            )
            st.stop()

        st.session_state["_data"]   = data
        st.session_state["_waktu"]  = datetime.datetime.combine(
            tanggal_input, datetime.time(0, 0)
        )
        st.session_state["_sumber"] = "csv"

d       = st.session_state["_data"]
ft      = st.session_state["_waktu"]
sumber  = st.session_state["_sumber"]

# ── Tampilkan komponen ISPU ───────────────────────────────────────────────────
st.divider()

if sumber == "aqicn":
    waktu_str = ft.strftime("%d %B %Y, %H:%M") + " WIB (Real-time)"
else:
    waktu_str = ft.strftime("%d %B %Y") + " (Data Historis CSV)"

st.subheader(f"📊 Data ISPU · {pilihan}")
st.caption(f"📡 Stasiun {stasiun} – {nama_stasiun} · {waktu_str} · {sumber_desc}")

cols = st.columns(6)
for col, key in zip(cols, FITUR):
    val   = d[key]
    batas = AMBANG[key]
    with col:
        st.metric(
            label       = f"{FITUR_DISPLAY[key]} (µg/m³)",
            value       = val,
            delta       = f"{val - batas:+d} dari batas {batas}",
            delta_color = "inverse",
        )

# ── Prediksi KNN ──────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"🔍 Hasil Analisis Kualitas Udara · {pilihan}")

input_arr    = np.array([[d[k] for k in FITUR]])
input_scaled = scaler.transform(input_arr)
prediksi     = model.predict(input_scaled)[0]
proba        = model.predict_proba(input_scaled)[0]
kelas        = meta["label_classes"]

melebihi_label = [FITUR_DISPLAY[k] for k in FITUR if d[k] > AMBANG[k]]

col_narasi, col_detail = st.columns([1, 1], gap="large")

# ── Kolom kiri: narasi ────────────────────────────────────────────────────────
with col_narasi:
    if prediksi == "Sehat":
        st.success(f"## ✅ {pilihan}: Kualitas Udara SEHAT")
        st.markdown(f"""
Analisis KNN terhadap data ISPU dari Stasiun **{nama_stasiun}** ({stasiun})
menunjukkan bahwa kawasan **{pilihan}** ({wilayah}) memiliki kualitas
udara yang **tergolong Sehat** pada **{waktu_str}**.

Kadar polutan berada dalam batas aman ISPU. Aktivitas luar ruangan
dapat dilakukan seperti biasa oleh seluruh kalangan masyarakat.
        """)
    else:
        st.error(f"## ⚠️ {pilihan}: Kualitas Udara TIDAK SEHAT")
        st.markdown(f"""
Analisis KNN terhadap data ISPU dari Stasiun **{nama_stasiun}** ({stasiun})
menunjukkan bahwa kawasan **{pilihan}** ({wilayah}) memiliki kualitas
udara yang **tergolong Tidak Sehat** pada **{waktu_str}**.

Tindakan yang disarankan untuk masyarakat sekitar:
- 😷 Gunakan masker N95/KN95 saat berada di luar ruangan
- 🏠 Batasi aktivitas luar, terutama untuk lansia dan anak-anak
- 🪟 Tutup ventilasi untuk mencegah polutan masuk ke dalam ruangan
        """)

    if melebihi_label:
        st.warning(f"🔴 Komponen melebihi ambang batas: **{', '.join(melebihi_label)}**")
    else:
        st.info("🟢 Semua komponen berada di bawah ambang batas aman.")

    # Badge sumber data
    if sumber == "aqicn":
        st.caption("📡 Data bersumber dari **API real-time AQICN**")
    else:
        st.caption("📂 Data bersumber dari **CSV historis ISPU DKI 2023**")

# ── Kolom kanan: probabilitas + tabel ─────────────────────────────────────────
with col_detail:
    st.markdown("#### 📈 Probabilitas Klasifikasi")
    for label, prob in zip(kelas, proba):
        icon = "✅" if label == "Sehat" else "⚠️"
        st.progress(float(prob), text=f"{icon} {label}: {prob*100:.1f}%")

    st.divider()
    st.markdown("#### 🗂️ Ringkasan Komponen ISPU")
    df_tbl = pd.DataFrame({
        "Komponen":      [FITUR_DISPLAY[k] for k in FITUR],
        "Nilai (µg/m³)": [d[k]             for k in FITUR],
        "Batas Aman":    [AMBANG[k]         for k in FITUR],
        "Status":        [badge(d[k], AMBANG[k]) for k in FITUR],
    })
    st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    st.caption(
        f"📡 {stasiun} ({nama_stasiun}) · {waktu_str} | "
        f"KNN k={meta['k_terbaik']} · Akurasi {meta['akurasi_test']*100:.1f}%"
    )

# ── Tren historis (jika CSV tersedia) ─────────────────────────────────────────
if not df_historis.empty:
    st.divider()
    st.subheader(f"📉 Tren Historis 2023 · {nama_stasiun} ({stasiun})")

    df_tren = df_historis[df_historis["stasiun_kode"] == stasiun].copy()
    df_tren = df_tren.sort_values("tanggal")

    fitur_pilih = st.multiselect(
        "Pilih komponen untuk ditampilkan:",
        options=FITUR,
        default=["PM10", "PM25"],
        format_func=lambda x: FITUR_DISPLAY[x],
    )

    if fitur_pilih:
        df_chart = df_tren.set_index("tanggal")[fitur_pilih]
        st.line_chart(df_chart, use_container_width=True)
    else:
        st.info("Pilih minimal satu komponen untuk melihat grafik tren.")