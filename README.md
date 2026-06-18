# SmartFlood ID (Surabaya Flood Prediction & Early Warning System)

SmartFlood ID merupakan sistem pendukung keputusan berbasis **Big Data dan Machine Learning** yang dirancang untuk melakukan prediksi risiko banjir serta memberikan peringatan dini pada wilayah **15 kecamatan di Kota Surabaya**.

Sistem ini mengintegrasikan model **Machine Learning**, arsitektur **data streaming real-time**, dan dashboard interaktif untuk memproses data cuaca serta kondisi hidrologi secara langsung guna membantu analisis potensi banjir.

---

## 👥 Anggota Tim

| No | Nama                           | NRP        |
| -- | ------------------------------ | ---------- |
| 1  | Salomo         | 502722063 |
| 2  | Nabilah Anindya Paramesti | 5027241006 |
| 3  | Muhammad Farrel Rafli Al Fasya                         | 5027221075 |
| 4  | Ivan Syarifuddin      | 5027241045 |
| 5  | Mohammad Abyan Ranuaji               | 5027241106 |

---

# 🛠️ Teknologi yang Digunakan

SmartFlood ID dibangun menggunakan ekosistem Python untuk kebutuhan data processing, machine learning, streaming, dan visualisasi.

### Programming Language

Python 3.11

### Machine Learning & Data Processing

* **Scikit-Learn** — Random Forest Classifier
* **Pandas** — Data manipulation
* **NumPy** — Numerical computation
* **Joblib** — Model serialization

### Data Streaming & Ingestion

* **Apache Kafka**
* **kafka-python**

### Dashboard & User Interface

* **Streamlit**
* **Streamlit-Autorefresh**

### Data Visualization

* **Matplotlib**
* **Seaborn**

---

# 📡 Sumber Data & API Terintegrasi

## 1. Open-Meteo API

Endpoint:

```
https://api.open-meteo.com
```

Digunakan oleh Kafka Producer untuk mengambil data cuaca Kota Surabaya secara real-time, meliputi:

* Suhu
* Kelembaban
* Curah hujan
* Kecepatan angin

---

## 2. PetaBencana.id

Endpoint:

```
https://data.petabencana.id/reports
```

Digunakan sebagai referensi format payload laporan warga dan simulasi data laporan banjir secara real-time (Mock API).

---

## 3. Dataset Historis BMKG & BPBD Surabaya

Dataset sekunder yang digunakan pada tahap training model:

* 527 hari data cuaca historis
* 30 hari data ground-truth kejadian banjir

Dataset ini digunakan untuk membangun dan mengevaluasi model Machine Learning pada tahap training.

---

# 🚀 Fitur Utama

## Real-Time Data Ingestion

Mengambil dan memproses data streaming menggunakan Apache Kafka dari:

* Data cuaca
* Simulasi laporan warga

---

## Flood Risk Prediction

Model Random Forest digunakan untuk memprediksi:

* Probabilitas terjadinya banjir
* Estimasi tinggi genangan
* Perkiraan jumlah warga terdampak

Prediksi didukung oleh parameter hidrologi seperti:

* Koefisien limpasan
* Kapasitas drainase
* Elevasi wilayah

---

## Lead Time Calculation

Menghitung estimasi waktu respons sebelum kondisi genangan mencapai tingkat kritis.

---

## Manual Simulation Mode

Dashboard menyediakan mode simulasi interaktif untuk menguji respons sistem terhadap skenario cuaca ekstrem tanpa perlu menunggu data streaming aktual.

---

# 📂 Struktur Project

```
SmartFlood-ID/
│
├── app.py                  # Streamlit Dashboard
├── producer_cuaca.py       # Kafka Producer data cuaca
├── producer_laporan.py     # Kafka Producer simulasi laporan warga
├── model/
│   └── smartflood_rf_model.pkl   # model utama
│   └── smartflood_encoder.pkl # encoder
├── requirements.txt        # Python dependencies
│
└── README.md
```

---

# ⚙️ Cara Menjalankan Lokal

Karena menggunakan arsitektur **event-driven**, pastikan Apache Kafka dan Zookeeper sudah aktif terlebih dahulu.

Default Kafka:

```
localhost:9092
```

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Jalankan Kafka Producer Cuaca

Terminal pertama:

```bash
python3 producer_cuaca.py
```

Producer akan mengambil data cuaca secara real-time dari Open-Meteo API.

---

## 3. Jalankan Kafka Producer Laporan Warga

Terminal kedua:

```bash
python3 producer_laporan.py
```

Producer ini menjalankan simulasi laporan banjir warga.

---

## 4. Jalankan Streamlit Dashboard

Terminal ketiga:

```bash
streamlit run app.py
```

Dashboard dapat diakses melalui:

```
http://localhost:8501
```

---

# 🌐 Deployment Cloud

SmartFlood ID juga telah tersedia dalam bentuk dashboard online menggunakan Streamlit Cloud.

Akses aplikasi:

[SmartFlood ID — Streamlit Cloud](https://smartflood-surabaya.streamlit.app/)

---

# 📌 Catatan

* Model Machine Learning dilatih menggunakan dataset historis sebelum dilakukan deployment.
* Sistem streaming menggunakan Kafka untuk mensimulasikan kondisi real-time.
* Dashboard digunakan sebagai media monitoring dan visualisasi hasil prediksi banjir.
