# Ringkasan Proyek: Prediksi Harga Beras Medium (Minggu 3 & 4)

Dokumen ini merangkum seluruh pencapaian teknis dan substansi akademik yang telah diselesaikan untuk memenuhi target Minggu 3 dan Minggu 4.

---

## 📅 Minggu 3: Eksperimen Pemrograman Model JST
**Fokus:** Membangun "otak" buatan menggunakan algoritma Backpropagation.

### 1. Implementasi Teknis
*   **Normalisasi (MinMaxScaler):** Mengubah harga beras (Rp 9.000 - Rp 15.000) menjadi rentang **0 hingga 1**. Hal ini wajib dilakukan agar fungsi aktivasi `Tanh` pada JST tidak mengalami *saturation* (jenuh) dan proses update bobot lebih stabil.
*   **Arsitektur JST:**
    *   **Input Layer:** 2 Neuron (menerima input harga 1 bulan lalu dan 2 bulan lalu).
    *   **Hidden Layer:** 2 Lapis (8 neuron di lapis pertama, 4 neuron di lapis kedua).
    *   **Output Layer:** 1 Neuron (memprediksi harga bulan berjalan).
*   **Algoritma:** Backpropagation dengan optimizer **Adam** dan fungsi aktivasi **Tanh**.
*   **Training-Testing Split:** Menggunakan rasio **80:20** tanpa *shuffle* (karena data bersifat *time-series*/urutan waktu).

### 2. Hasil Luaran
*   **Loss Curve:** Grafik yang menunjukkan penurunan error seiring bertambahnya iterasi (*epoch*).
*   **File:** `model_training.ipynb`.

---

## 📅 Minggu 4: Evaluasi Mendalam & Analisis Hasil
**Fokus:** Menguji seberapa pintar model dan menganalisis kesalahannya.

### 1. Pengukuran Kinerja (Metrik Regresi)
*   **MSE (Mean Squared Error):** Menghitung rata-rata kuadrat kesalahan. Memberikan hukuman berat pada error yang besar.
*   **RMSE (Root Mean Squared Error):** Akar dari MSE. Nilai ini menunjukkan rata-rata selisih prediksi dalam satuan asli (**Rupiah**).
*   **MAE (Mean Absolute Error):** Menunjukkan rata-rata absolut selisih harga. Ini adalah angka paling mudah dijelaskan ke dosen (Misal: "Rata-rata error model adalah Rp 200").

### 2. Analisis Visual
*   **Scatter Plot (Aktual vs Prediksi):** Jika titik-titik berada di sekitar garis diagonal, model memiliki korelasi yang sangat kuat antara kenyataan dan prediksi.
*   **Residual Distribution:** Melihat apakah error berpusat di angka 0. Jika iya, model tidak bias (adil dalam memprediksi harga tinggi maupun rendah).

---

## 🎓 Poin Penting untuk Paparan/Ujian (Defense)

### Mengapa menggunakan JST Backpropagation?
> "Karena JST Backpropagation sangat unggul dalam mengenali pola non-linear pada data deret waktu (*time series*). Algoritma ini secara iteratif memperbaiki bobot internalnya berdasarkan selisih error (Backpro), sehingga akurasi prediksi terus meningkat."

### Apa arti nilai RMSE/MAE Anda?
> "Nilai MAE sebesar Rp [Lihat Hasil Anda] menunjukkan bahwa model kami secara rata-rata hanya meleset sebesar nominal tersebut dari harga pasar sebenarnya di Jawa Timur. Ini menunjukkan tingkat presisi yang tinggi untuk perencanaan ekonomi daerah."

### Mengapa ada error (Residual)?
> "Error terjadi karena fluktuasi harga pangan tidak hanya dipengaruhi oleh harga masa lalu, tetapi juga faktor eksternal seperti musim panen, kebijakan subsidi, atau hambatan distribusi yang datanya tidak masuk dalam variabel input model."

---

## 📁 Daftar File Final
1.  `preprocess.ipynb` (Minggu 2)
2.  `model_training.ipynb` (Minggu 3)
3.  `evaluation_analysis.ipynb` (Minggu 4)
4.  `README.md` (Dokumentasi Repository)
5.  `beras_medium_monthly.csv` (Data Olahan)
6.  `prediction_results.csv` (Hasil Prediksi untuk Evaluasi)
