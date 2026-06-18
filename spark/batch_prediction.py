import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession

# 1. Inisialisasi Apache Spark
spark = SparkSession.builder \
    .appName("SmartFlood_Spark_Analytics") \
    .master("local[*]") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ SparkSession berhasil dibuat!")

# ==========================================
# ANALISIS 1: BACA DATA DARI HDFS & SPARK SQL
# ==========================================
try:
    # Spark akan membaca semua file JSON di folder ini sekaligus
    df_cuaca = spark.read.json("hdfs://localhost:8020/data/smartflood/cuaca/*.json")
    df_cuaca.createOrReplaceTempView("cuaca_view")
    
    print(f"✅ Data cuaca dibaca dari HDFS: {df_cuaca.count()} records.")

    # Ambil kondisi cuaca terbaru menggunakan Spark SQL
    latest_weather = spark.sql("""
        SELECT curah_hujan_mm, kelembaban_pct, suhu_c, kecepatan_angin_ms, timestamp
        FROM cuaca_view
        ORDER BY timestamp DESC
        LIMIT 1
    """).collect()[0]

    rr = latest_weather['curah_hujan_mm']
    rh = latest_weather['kelembaban_pct']
    tavg = latest_weather['suhu_c']
    wind = latest_weather['kecepatan_angin_ms']
    latest_time = latest_weather['timestamp']

except Exception as e:
    print("⚠️ Data di HDFS belum siap atau kosong. Memakai data default.", e)
    rr, rh, tavg, wind, latest_time = 15.0, 80, 28.0, 5.0, "N/A"

# ==========================================
# ANALISIS 2: PREDIKSI RISIKO 15 KECAMATAN
# ==========================================
HIDROLOGI = [
    {'kecamatan': 'Benowo', 'elevasi_m': 3.2, 'kapasitas_drainase_m3s': 15, 'luas_km2': 24.8, 'penduduk': 72541, 'koef_limpasan': 0.65, 'jarak_pantai_km': 12.5},
    {'kecamatan': 'Pakal', 'elevasi_m': 4.1, 'kapasitas_drainase_m3s': 18, 'luas_km2': 22.5, 'penduduk': 63218, 'koef_limpasan': 0.60, 'jarak_pantai_km': 15.2},
    {'kecamatan': 'Tandes', 'elevasi_m': 5.0, 'kapasitas_drainase_m3s': 22, 'luas_km2': 9.7, 'penduduk': 108432, 'koef_limpasan': 0.72, 'jarak_pantai_km': 8.3},
    {'kecamatan': 'Lakarsantri', 'elevasi_m': 8.5, 'kapasitas_drainase_m3s': 35, 'luas_km2': 18.4, 'penduduk': 57819, 'koef_limpasan': 0.45, 'jarak_pantai_km': 20.1},
    {'kecamatan': 'Wonokromo', 'elevasi_m': 6.2, 'kapasitas_drainase_m3s': 28, 'luas_km2': 8.5, 'penduduk': 143265, 'koef_limpasan': 0.80, 'jarak_pantai_km': 9.8},
    {'kecamatan': 'Rungkut', 'elevasi_m': 9.8, 'kapasitas_drainase_m3s': 40, 'luas_km2': 21.1, 'penduduk': 95847, 'koef_limpasan': 0.70, 'jarak_pantai_km': 18.5},
    {'kecamatan': 'Sukolilo', 'elevasi_m': 7.3, 'kapasitas_drainase_m3s': 32, 'luas_km2': 23.5, 'penduduk': 87631, 'koef_limpasan': 0.65, 'jarak_pantai_km': 14.2},
    {'kecamatan': 'Kenjeran', 'elevasi_m': 2.8, 'kapasitas_drainase_m3s': 12, 'luas_km2': 7.6, 'penduduk': 75234, 'koef_limpasan': 0.75, 'jarak_pantai_km': 1.2},
    {'kecamatan': 'Bulak', 'elevasi_m': 2.5, 'kapasitas_drainase_m3s': 10, 'luas_km2': 6.8, 'penduduk': 42187, 'koef_limpasan': 0.78, 'jarak_pantai_km': 0.8},
    {'kecamatan': 'Semampir', 'elevasi_m': 3.5, 'kapasitas_drainase_m3s': 14, 'luas_km2': 8.9, 'penduduk': 132541, 'koef_limpasan': 0.82, 'jarak_pantai_km': 2.1},
    {'kecamatan': 'Bubutan', 'elevasi_m': 4.2, 'kapasitas_drainase_m3s': 20, 'luas_km2': 4.8, 'penduduk': 98765, 'koef_limpasan': 0.85, 'jarak_pantai_km': 6.5},
    {'kecamatan': 'Simokerto', 'elevasi_m': 4.8, 'kapasitas_drainase_m3s': 18, 'luas_km2': 5.3, 'penduduk': 112340, 'koef_limpasan': 0.83, 'jarak_pantai_km': 4.3},
    {'kecamatan': 'Sawahan', 'elevasi_m': 5.1, 'kapasitas_drainase_m3s': 24, 'luas_km2': 9.1, 'penduduk': 187654, 'koef_limpasan': 0.78, 'jarak_pantai_km': 7.8},
    {'kecamatan': 'Genteng', 'elevasi_m': 6.0, 'kapasitas_drainase_m3s': 26, 'luas_km2': 4.3, 'penduduk': 76543, 'koef_limpasan': 0.80, 'jarak_pantai_km': 8.9},
    {'kecamatan': 'Gubeng', 'elevasi_m': 5.5, 'kapasitas_drainase_m3s': 30, 'luas_km2': 7.8, 'penduduk': 145231, 'koef_limpasan': 0.75, 'jarak_pantai_km': 10.2}
]

results = []
pasang_base = 0.8

print(f"\n⚙️ Memproses prediksi untuk curah hujan: {rr} mm...")
for kec in HIDROLOGI:
    elev = kec['elevasi_m']
    kap = kec['kapasitas_drainase_m3s']
    koef = kec['koef_limpasan']
    jarak = kec['jarak_pantai_km']
    luas = kec['luas_km2']
    pddk = kec['penduduk']

    tma = np.clip(rr/25*(1+(5-elev)/5), 0.2, 5).round(2)
    pasang = max(0, pasang_base - jarak*0.05)
    debit_lr = koef*(rr/3600)*luas*1000
    excess = max(0, debit_lr-kap)
    tinggi_gen = np.clip((excess/kap)*0.5+pasang*0.3+(5-elev)*0.1, 0, 3).round(2)
    luas_td = np.clip(luas*(tinggi_gen/2)*0.6, 0, luas).round(2)
    ptd = int((pddk/luas)*luas_td)
    lead = max(1, round(12-(rr/20)-(5-elev)*0.5))

    # Simulasi skor probabilitas
    indeks = rr*0.25 + tma*0.20 + (10-elev)*0.15 + excess*0.15 + pasang*0.10
    prob = min(100.0, (indeks / 5.0) * 100)
    status = '🔴 KRITIS' if prob>70 else '🟡 WASPADA' if prob>40 else '🟢 AMAN'

    results.append({
        'Kecamatan': kec['kecamatan'],
        'Status': status,
        'Probabilitas': round(prob, 1),
        'Tinggi_Genangan_m': float(tinggi_gen),
        'Penduduk_Terdampak': ptd,
        'Lead_Time_jam': lead
    })

df_results = pd.DataFrame(results).sort_values('Probabilitas', ascending=False)

# ==========================================
# SIMPAN HASIL KE LOKAL (UNTUK STREAMLIT) & HDFS
# ==========================================
output_payload = {
    "timestamp_analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "cuaca_referensi": {
        "waktu": latest_time,
        "curah_hujan": float(rr),
        "suhu": float(tavg),
        "angin": float(wind)
    },
    "prediksi_kecamatan": df_results.to_dict(orient='records')
}

# Simpan ke folder lokal
os.makedirs("dashboard/data", exist_ok=True)
with open("dashboard/data/spark_results.json", "w") as f:
    json.dump(output_payload, f, indent=4)
print("✅ Hasil prediksi tersimpan di lokal (dashboard/data/spark_results.json)")

# Simpan ke HDFS
try:
    df_spark_res = spark.createDataFrame(df_results)
    df_spark_res.write.mode("overwrite").json("hdfs://localhost:8020/data/smartflood/hasil/prediksi_latest")
    print("✅ Hasil prediksi tersimpan di HDFS (/data/smartflood/hasil/prediksi_latest)")
except Exception as e:
    print("⚠️ Gagal menyimpan output ke HDFS.", e)

spark.stop()
