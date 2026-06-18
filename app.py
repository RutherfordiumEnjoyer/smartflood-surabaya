import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import json
import os
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SmartFlood ID — Surabaya",
    page_icon="🌊",
    layout="wide"
)

# --- LOAD CUSTOM CSS ---
def load_css(path: str):
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- PALETTE CHART MATPLOTLIB (Warna Pasir) ---
PALETTE = {
    "bg": "#F4F0EA",       # <-- Berubah jadi warna pasir
    "primary": "#0A3A52",
    "text": "#334155",
    "grid": "#D5C9B8",     # Grid line warna pasir gelap
    "safe": "#10B981",
    "warn": "#F59E0B",
    "danger": "#EF4444",
}

mpl.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.labelcolor": PALETTE["text"],
    "text.color": PALETTE["primary"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
})

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE["grid"])
    ax.spines['bottom'].set_color(PALETTE["grid"])
    ax.grid(axis='x', color=PALETTE["grid"], linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

# --- LOAD MACHINE LEARNING MODEL (UNSCALED) ---
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('smartflood_rf_model.pkl')
        le = joblib.load('smartflood_encoder.pkl')
        return rf, le
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

rf, le = load_models()

FEATURES = [
    'curah_hujan_mm', 'curah_hujan_3hari', 'kelembaban_pct', 'suhu_c',
    'kecepatan_angin', 'durasi_hujan_jam', 'bulan', 'musim_hujan',
    'elevasi_m', 'koef_limpasan', 'frekuensi_historis', 'kecamatan_enc'
]

HIDROLOGI = pd.DataFrame({
    'kecamatan': ['Benowo','Pakal','Tandes','Lakarsantri','Wonokromo',
                  'Rungkut','Sukolilo','Kenjeran','Bulak','Semampir',
                  'Bubutan','Simokerto','Sawahan','Genteng','Gubeng'],
    'elevasi_m': [3.2,4.1,5.0,8.5,6.2,9.8,7.3,2.8,2.5,3.5,4.2,4.8,5.1,6.0,5.5],
    'kapasitas_drainase_m3s': [15,18,22,35,28,40,32,12,10,14,20,18,24,26,30],
    'luas_km2': [24.8,22.5,9.7,18.4,8.5,21.1,23.5,7.6,6.8,8.9,4.8,5.3,9.1,4.3,7.8],
    'penduduk': [72541,63218,108432,57819,143265,95847,87631,75234,42187,
                 132541,98765,112340,187654,76543,145231],
    'koef_limpasan': [0.65,0.60,0.72,0.45,0.80,0.70,0.65,0.75,0.78,0.82,0.85,0.83,0.78,0.80,0.75],
    'jarak_pantai_km': [12.5,15.2,8.3,20.1,9.8,18.5,14.2,1.2,0.8,2.1,6.5,4.3,7.8,8.9,10.2],
    'frekuensi_historis': [8,6,10,3,12,4,5,15,18,14,9,11,10,7,8]
})

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try: return json.load(f)
            except: return None
    return None

live_cuaca = load_json('dashboard/data/live_cuaca.json')
live_laporan = load_json('dashboard/data/live_laporan.json')

# --- HEADER UI ---
# Wave memotong menjadi warna pasir pantai (#F4F0EA)
WAVE_SVG = """
<svg class="wave-divider" viewBox="0 0 1440 60" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M0,32 C240,70 480,0 720,28 C960,56 1200,8 1440,30 L1440,60 L0,60 Z" fill="#F4F0EA"></path>
</svg>
"""
st.markdown(f"""
<div class="flood-header">
  <h1>🌊 SmartFlood ID</h1>
  <p>Sistem Prediksi &amp; Peringatan Dini Banjir Berbasis Big Data — Kota Surabaya</p>
  {WAVE_SVG}
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🔄 Mode Sistem")
    app_mode = st.radio("Sumber Ingestion Data:", ["📡 Real-Time (Kafka Stream)", "🎛️ Simulasi Manual"])
    
    if app_mode == "📡 Real-Time (Kafka Stream)":
        st_autorefresh(interval=20000, limit=None, key="kafka_refresh")
    
    st.markdown("---")
    
    if app_mode == "🎛️ Simulasi Manual":
        st.markdown("### ⚙️ Input Cuaca Manual")
        rr = st.slider("Curah hujan (mm/hari)", 0, 200, 45)
        rr_3d = st.slider("Akumulasi Hujan 3-hari (mm)", 0, 300, 75)
        rh = st.slider("Kelembaban (%)", 55, 100, 85)
        tavg = st.slider("Suhu rata-rata (°C)", 24.0, 36.0, 28.5)
        wind = st.slider("Kecepatan angin (m/s)", 1.0, 30.0, 5.0)
        ss = st.slider("Lama penyinaran (jam)", 0.0, 12.0, 3.0)
        bulan = st.selectbox("Bulan", list(range(1,13)), index=11)
    else:
        st.success("Sistem mendengarkan Apache Kafka...")
        st.caption("Prediksi berjalan otomatis dengan payload JSON terbaru.")

# --- PIPELINE PREDIKSI ---
if app_mode == "📡 Real-Time (Kafka Stream)":
    if not live_cuaca:
        st.warning("⚠️ Belum ada data dari Kafka (`live_cuaca.json`).")
        st.stop()
    else:
        rr = live_cuaca.get('curah_hujan_mm', 0)
        rh = live_cuaca.get('kelembaban_pct', 80)
        tavg = live_cuaca.get('suhu_c', 28.0)
        wind = live_cuaca.get('kecepatan_angin_ms', 3.0)
        rr_3d = rr * 1.5 + 20 
        ss = 4.0 if rr > 10 else 8.0
        bulan = datetime.now().month
        timestamp_text = f"Live Kafka Data: {live_cuaca.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}"
else:
    timestamp_text = "Mode Simulasi Manual (Local ML)"

musim = 1 if bulan in [11,12,1,2,3,4] else 0
durasi = max(0, round((8-ss)*min(rr/20,1)))
pasang_base = 0.8 + 0.4*np.sin(2*np.pi*bulan/12)

results = []
for _, kec in HIDROLOGI.iterrows():
    debit_lr = kec['koef_limpasan']*(rr/3600)*kec['luas_km2']*1000
    excess = max(0, debit_lr - kec['kapasitas_drainase_m3s'])
    pasang = max(0, pasang_base - kec['jarak_pantai_km']*0.05)
    
    tinggi_gen = np.clip((excess/kec['kapasitas_drainase_m3s'])*0.5 + pasang*0.3 + (5-kec['elevasi_m'])*0.1, 0.05, 3).round(2)
    luas_td = np.clip(kec['luas_km2']*(tinggi_gen/2)*0.6, 0.1, kec['luas_km2']).round(2)
    ptd = int((kec['penduduk']/kec['luas_km2']) * luas_td)
    lead = max(1, round(12-(rr/20)-(5-kec['elevasi_m'])*0.5))
    
    feat_dict = {
        'curah_hujan_mm': rr, 'curah_hujan_3hari': rr_3d, 'kelembaban_pct': rh,
        'suhu_c': tavg, 'kecepatan_angin': wind, 'durasi_hujan_jam': durasi,
        'bulan': bulan, 'musim_hujan': musim, 'elevasi_m': kec['elevasi_m'],
        'koef_limpasan': kec['koef_limpasan'], 'frekuensi_historis': kec['frekuensi_historis'],
        'kecamatan_enc': le.transform([kec['kecamatan']])[0]
    }
    
    df_feat = pd.DataFrame([feat_dict])[FEATURES]
    prob = rf.predict_proba(df_feat)[0][1] * 100
    
    status = '🔴 KRITIS' if prob > 70 else '🟡 WASPADA' if prob > 40 else '🟢 AMAN'

    # --- Komponen tambahan untuk Flood Risk Score komposit ---
    pct_luas = float(np.clip((luas_td / kec['luas_km2']) * 100, 0, 100))          # % wilayah kecamatan yang tergenang
    pct_penduduk = float(np.clip((ptd / kec['penduduk']) * 100, 0, 100))          # % populasi kecamatan yang terdampak
    norm_genangan = float(np.clip((tinggi_gen / 3) * 100, 0, 100))                # tinggi genangan dinormalisasi ke skala 0-100 (cap 3m)
    norm_lead = float(np.clip(((12 - lead) / 11) * 100, 0, 100))                  # makin pendek lead time -> makin urgent

    # Bobot bersifat asumsi awal (belum dikalibrasi/divalidasi), perlu sensitivity test sebelum dipakai operasional
    risk_score = round(
        0.35 * prob +
        0.20 * norm_genangan +
        0.15 * pct_luas +
        0.20 * pct_penduduk +
        0.10 * norm_lead,
        1
    )
    prioritas = '🔴 PRIORITAS UTAMA' if risk_score > 70 else '🟡 PRIORITAS SEDANG' if risk_score > 40 else '🟢 PRIORITAS RENDAH'

    results.append({
        'Kecamatan': kec['kecamatan'], 'Status': status, 'Probabilitas (%)': prob,
        'Tinggi Genangan (m)': tinggi_gen, 'Luas Terdampak (km2)': luas_td,
        '% Luas Terdampak': round(pct_luas, 1), 'Penduduk Terdampak': ptd,
        'Lead Time (jam)': lead, 'Flood Risk Score': risk_score, 'Prioritas Evakuasi': prioritas
    })

df_pred = pd.DataFrame(results).sort_values('Probabilitas (%)', ascending=False)

# --- METRIC CARDS ---
kritis = (df_pred['Probabilitas (%)'] > 70).sum()
waspada = ((df_pred['Probabilitas (%)'] > 40) & (df_pred['Probabilitas (%)'] <= 70)).sum()
total_ptd = df_pred['Penduduk Terdampak'].sum()
min_lead = df_pred['Lead Time (jam)'].min()

st.markdown(f'<div class="status-badge">📡 <span>{timestamp_text} &nbsp;|&nbsp; Curah Hujan: <b>{rr} mm</b></span></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    cls = 'danger' if kritis > 0 else 'safe'
    st.markdown(f'<div class="metric-card {cls}"><div class="card-label">Kecamatan Kritis (&gt;70%)</div><div class="big-num">{kritis}</div></div>', unsafe_allow_html=True)
with c2:
    cls = 'warn' if waspada > 0 else 'safe'
    st.markdown(f'<div class="metric-card {cls}"><div class="card-label">Kecamatan Waspada (&gt;40%)</div><div class="big-num">{waspada}</div></div>', unsafe_allow_html=True)
with c3:
    cls = 'danger' if total_ptd > 5000 else 'warn' if total_ptd > 1000 else 'safe'
    st.markdown(f'<div class="metric-card {cls}"><div class="card-label">Estimasi Terdampak</div><div class="big-num">{total_ptd:,} <span style="font-size:1rem;font-weight:normal">jiwa</span></div></div>', unsafe_allow_html=True)
with c4:
    cls = 'danger' if min_lead <= 3 else 'warn' if min_lead <= 6 else 'safe'
    st.markdown(f'<div class="metric-card {cls}"><div class="card-label">Lead Time Tercepat</div><div class="big-num">{min_lead} <span style="font-size:1rem;font-weight:normal">jam</span></div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Analisis Prediksi", "📡 Event Kafka API", "📋 Detail Tabel", "📊 Evaluasi Model", "🌦️ Data BMKG"])

with tab1:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Probabilitas
    df_sorted = df_pred.sort_values('Probabilitas (%)')
    bc = [PALETTE["danger"] if p > 70 else PALETTE["warn"] if p > 40 else PALETTE["safe"] for p in df_sorted['Probabilitas (%)']]
    axes[0,0].barh(df_sorted['Kecamatan'], df_sorted['Probabilitas (%)'], color=bc)
    axes[0,0].axvline(70, color=PALETTE["danger"], linestyle='--', alpha=0.8, label='Threshold Kritis (70%)')
    axes[0,0].axvline(40, color=PALETTE["warn"], linestyle='--', alpha=0.8, label='Threshold Waspada (40%)')
    axes[0,0].set_title('Probabilitas Banjir (%)')
    axes[0,0].set_xlim(0, 105)
    axes[0,0].legend(frameon=False)
    style_axes(axes[0,0])

    # 2. Genangan
    tg_sorted = df_pred.sort_values('Tinggi Genangan (m)')
    gc = [PALETTE["danger"] if g > 1.0 else PALETTE["warn"] if g > 0.5 else PALETTE["safe"] for g in tg_sorted['Tinggi Genangan (m)']]
    axes[0,1].barh(tg_sorted['Kecamatan'], tg_sorted['Tinggi Genangan (m)'], color=gc)
    axes[0,1].set_title('Estimasi Tinggi Genangan (m)')
    style_axes(axes[0,1])

    # 3. Terdampak
    pt_sorted = df_pred.sort_values('Penduduk Terdampak')
    pc = [PALETTE["danger"] if p > 5000 else PALETTE["warn"] if p > 1000 else PALETTE["safe"] for p in pt_sorted['Penduduk Terdampak']]
    axes[1,0].barh(pt_sorted['Kecamatan'], pt_sorted['Penduduk Terdampak'], color=pc)
    axes[1,0].set_title('Estimasi Penduduk Terdampak (jiwa)')
    style_axes(axes[1,0])

    # 4. Lead Time
    lt_sorted = df_pred.sort_values('Lead Time (jam)')
    lc = [PALETTE["danger"] if l <= 3 else PALETTE["warn"] if l <= 6 else PALETTE["safe"] for l in lt_sorted['Lead Time (jam)']]
    axes[1,1].barh(lt_sorted['Kecamatan'], lt_sorted['Lead Time (jam)'], color=lc)
    axes[1,1].axvline(3, color=PALETTE["danger"], linestyle='--', alpha=0.8, label='Kritis (≤3 jam)')
    axes[1,1].axvline(6, color=PALETTE["warn"], linestyle='--', alpha=0.8, label='Waspada (≤6 jam)')
    axes[1,1].set_title('Lead Time Peringatan Dini (jam)')
    axes[1,1].legend(frameon=False)
    style_axes(axes[1,1])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### 🎯 Flood Risk Score, Luas Terdampak, & Prioritas Evakuasi")
    st.caption("Flood Risk Score = skor komposit (probabilitas, tinggi genangan, % luas terdampak, % populasi terdampak, urgensi lead time). Bobot saat ini bersifat asumsi awal dan belum dikalibrasi terhadap data historis — perlu sensitivity analysis sebelum dipakai untuk keputusan operasional.")

    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))

    # 5. Luas Wilayah Terdampak (% dari luas kecamatan)
    lt_area_sorted = df_pred.sort_values('% Luas Terdampak')
    ac = [PALETTE["danger"] if a > 50 else PALETTE["warn"] if a > 25 else PALETTE["safe"] for a in lt_area_sorted['% Luas Terdampak']]
    axes2[0].barh(lt_area_sorted['Kecamatan'], lt_area_sorted['% Luas Terdampak'], color=ac)
    axes2[0].axvline(50, color=PALETTE["danger"], linestyle='--', alpha=0.8)
    axes2[0].axvline(25, color=PALETTE["warn"], linestyle='--', alpha=0.8)
    axes2[0].set_title('% Luas Wilayah Terdampak')
    axes2[0].set_xlim(0, 105)
    style_axes(axes2[0])

    # 6. Flood Risk Score komposit
    rs_sorted = df_pred.sort_values('Flood Risk Score')
    rc = [PALETTE["danger"] if r > 70 else PALETTE["warn"] if r > 40 else PALETTE["safe"] for r in rs_sorted['Flood Risk Score']]
    axes2[1].barh(rs_sorted['Kecamatan'], rs_sorted['Flood Risk Score'], color=rc)
    axes2[1].axvline(70, color=PALETTE["danger"], linestyle='--', alpha=0.8, label='Prioritas Utama (>70)')
    axes2[1].axvline(40, color=PALETTE["warn"], linestyle='--', alpha=0.8, label='Prioritas Sedang (>40)')
    axes2[1].set_title('Flood Risk Score (Komposit)')
    axes2[1].set_xlim(0, 105)
    axes2[1].legend(frameon=False, fontsize=8)
    style_axes(axes2[1])

    # 7. Top 5 Prioritas Evakuasi/Penanganan
    top5 = df_pred.sort_values('Flood Risk Score', ascending=False).head(5).sort_values('Flood Risk Score')
    axes2[2].barh(top5['Kecamatan'], top5['Flood Risk Score'], color=PALETTE["danger"])
    for i, (_, row) in enumerate(top5.iterrows()):
        axes2[2].text(
            row['Flood Risk Score'] + 2, i,
            f"{row['Penduduk Terdampak']:,} jiwa | {row['% Luas Terdampak']:.0f}% wilayah | {row['Lead Time (jam)']} jam",
            va='center', fontsize=8, color=PALETTE["text"]
        )
    axes2[2].set_title('Top 5 Prioritas Evakuasi/Penanganan')
    axes2[2].set_xlim(0, 140)
    style_axes(axes2[2])

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("#### 🚨 Rekomendasi Wilayah Prioritas")
    top3 = df_pred.sort_values('Flood Risk Score', ascending=False).head(3)
    rec_lines = []
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        rec_lines.append(
            f"**{rank}. {row['Kecamatan']}** — Flood Risk Score `{row['Flood Risk Score']:.1f}` ({row['Prioritas Evakuasi']})  \n"
            f"Genangan estimasi **{row['Tinggi Genangan (m)']:.2f} m**, **{row['% Luas Terdampak']:.0f}%** luas wilayah terendam, "
            f"**{row['Penduduk Terdampak']:,} jiwa** terdampak, lead time peringatan **{row['Lead Time (jam)']} jam**."
        )
    st.markdown("\n\n".join(rec_lines))
    st.caption("Rekomendasi berbasis ranking Flood Risk Score saat ini, bukan rencana evakuasi resmi. Validasi lapangan oleh BPBD tetap diperlukan sebelum eksekusi.")

with tab2:
    st.markdown("### 📡 Ingestion Layer: Data dari Apache Kafka")
    st.caption("Monitoring aliran data real-time yang masuk ke dalam antrean (broker) Kafka.")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.subheader("🌦️ Event Cuaca (Open-Meteo)")
        if live_cuaca: st.json(live_cuaca)
        else: st.info("Menunggu topic `smartflood-cuaca`...")
    with col_k2:
        st.subheader("📰 Laporan Warga (RSS/Sosmed)")
        if live_laporan: st.json(live_laporan)
        else: st.info("Menunggu topic `smartflood-laporan`...")

with tab3:
    st.markdown("### 📋 Tabel Rincian per Kecamatan")
    st.dataframe(
        df_pred, use_container_width=True, hide_index=True,
        column_config={
            "Probabilitas (%)": st.column_config.NumberColumn("Probabilitas (%)", format="%.2f %%"),
            "Tinggi Genangan (m)": st.column_config.NumberColumn("Tinggi Genangan (m)", format="%.2f m"),
            "Luas Terdampak (km2)": st.column_config.NumberColumn("Luas Terdampak (km²)", format="%.2f km²"),
            "% Luas Terdampak": st.column_config.NumberColumn("% Luas Terdampak", format="%.1f %%"),
            "Penduduk Terdampak": st.column_config.NumberColumn("Penduduk Terdampak", format="%d jiwa"),
            "Flood Risk Score": st.column_config.NumberColumn("Flood Risk Score", format="%.1f")
        }
    )

with tab4:
    st.markdown("### 📊 Evaluasi Model Random Forest")
    st.markdown("""
    **Tentang Validasi Ini:**
    * Label banjir didasarkan pada **curah hujan kumulatif 3 hari > 80mm + ground truth historis BPBD (30 tanggal)**.
    * `tma_sungai_m` dan `excess_drainase` berfungsi murni sebagai fitur prediksi (X).
    * Validasi menggunakan metode **5-Fold Time-Series Cross Validation (Unscaled)** untuk menjaga urutan waktu.
    """)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC Score", "0.7511")
    m2.metric("Akurasi", "93.00%")
    m3.metric("F1-Score (Banjir)", "0.31")
    m4.metric("Ground Truth", "30 Hari")
    
    st.markdown("---")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Hasil 5-Fold Cross Validation**")
        try: st.image("aug.png", use_container_width=True)
        except: st.warning("Gambar 'aug.png' tidak ditemukan.")
    with col_img2:
        st.markdown("**Feature Importance (Kontribusi Fitur)**")
        try: st.image("importance.png", use_container_width=True)
        except: st.warning("Gambar 'importance.png' tidak ditemukan.")

    st.markdown("---")
    st.markdown("#### 📈 Hasil Evaluasi Time-Series CV & Test Set")
    st.markdown("**Threshold optimal (F1.5-Score dari Time-Series CV): `0.465`**")
    st.code("""
=======================================================
  VALIDASI 5-FOLD TIME-SERIES CV (UNSCALED)
=======================================================
  AUC      per fold : [0.8967, 0.6292, 0.6877, 0.9029, 0.7479]
  AUC mean ± std    : 0.7729 ± 0.1102
  Accuracy mean     : 0.9397 ± 0.0272
  F1-Score mean     : 0.4090 ± 0.2129
=======================================================

=======================================================
TUNED RANDOM FOREST (Unscaled) — AUC: 0.7511
Evaluasi akhir di test set berurut-waktu (dengan threshold F1.5-Score):
              precision    recall  f1-score   support

Tidak Banjir       0.97      0.96      0.96      1510
      Banjir       0.27      0.35      0.31        71

    accuracy                           0.93      1581
   macro avg       0.62      0.65      0.63      1581
weighted avg       0.94      0.93      0.93      1581
    """, language="text")

with tab5:
    st.markdown("### 🌦️ Eksplorasi Dataset Training")
    st.markdown("**Data Real BMKG Stasiun Juanda**")
    st.caption("Periode: 2024-11-24 s/d 2026-05-04 | 527 hari")
    try: st.image("data.png", use_container_width=True)
    except: st.warning("Gambar 'data.png' tidak ditemukan.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("Total hari", "527")
    col_d2.metric("Hari hujan", "459")
    col_d3.metric("Max hujan", "87.4 mm")
    col_d4.metric("Ground truth", "30 hari banjir")
    st.markdown("---")
    st.markdown("Data di atas merupakan dataset historis yang digunakan untuk melatih (training) model Machine Learning di Google Colab. Model dilatih menggunakan 527 baris data cuaca harian yang kemudian dikombinasikan dengan data hidrologi statis dari 15 kecamatan di Surabaya, menghasilkan total ribuan rekaman komputasi untuk memastikan akurasi prediksi.")

st.markdown('<div class="flood-footer"><span>🌊 SmartFlood ID — Sistem Pendukung Keputusan, bukan pengganti peringatan resmi BPBD/BMKG.</span></div>', unsafe_allow_html=True)
