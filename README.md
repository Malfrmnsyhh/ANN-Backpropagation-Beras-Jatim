# Prediksi Fluktuasi Harga Beras Medium di Jawa Timur Menggunakan JST Backpropagation

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![JST](https://img.shields.io/badge/Method-ANN%20Backpropagation-orange.svg)
![Status](https://img.shields.io/badge/Status-Development-green.svg)

Repositori ini berisi implementasi Jaringan Saraf Tiruan (JST) dengan algoritma **Backpropagation** untuk memprediksi fluktuasi harga beras medium di wilayah Jawa Timur. Proyek ini merupakan bagian dari tugas akhir mata kuliah Artificial Intelligence 2026.

## 📌 Deskripsi Proyek
Model ini dikembangkan untuk membantu analisis tren harga pangan dengan menggunakan data historis (Time Series). Dengan memanfaatkan fitur *lag* (harga bulan sebelumnya), model belajar mengenali pola fluktuasi harga guna memberikan estimasi harga di masa mendatang.

## 🛠️ Alur Kerja (Workflows)
1. **Data Preprocessing**: Pembersihan data, penanganan outlier, agregasi bulanan, dan pembuatan fitur lag.
2. **Feature Engineering**: Normalisasi data menggunakan `MinMaxScaler` (0-1).
3. **Model Development**: Implementasi `MLPRegressor` dengan arsitektur 2 hidden layers.
4. **Evaluation**: Pengukuran kinerja menggunakan metrik regresi MSE, RMSE, dan MAE.

## 📁 Struktur Direktori
- `harga_pertanian.csv`: Dataset mentah fluktuasi harga pertanian.
- `preprocess.ipynb`: Notebook untuk pembersihan dan persiapan data.
- `model_training.ipynb`: Notebook untuk proses training model JST.
- `evaluation_analysis.ipynb`: Notebook untuk analisis error dan visualisasi hasil.
- `beras_medium_monthly.csv`: Dataset hasil preprocessing (siap pakai).

## 🚀 Cara Menggunakan
1. Clone repositori ini:
   ```bash
   git clone https://github.com/Malfrmnsyhh/ANN-Backpropagation-Beras-Jatim.git
   ```
2. Instal dependensi yang diperlukan:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Jalankan notebook sesuai urutan (Preprocess -> Training -> Evaluation).

## 📊 Hasil Evaluasi
Model dievaluasi menggunakan metrik Regresi:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

## ✍️ Penulis
- **Nama** - *Muhammad Akmal Firmansyah*
- Kelompok 4 - Kelas Paralel B

---
*Proyek ini mengikuti standar artikel ilmiah IEEE/APA.*
