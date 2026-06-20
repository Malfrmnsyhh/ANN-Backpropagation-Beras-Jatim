import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Prediksi Harga Beras Jatim", layout="wide")

st.title("🌾 Dashboard Prediksi Harga Beras Medium - Jawa Timur")
st.markdown("""
Dashboard ini menampilkan hasil analisis dan prediksi fluktuasi harga beras medium di Jawa Timur menggunakan 
**Jaringan Saraf Tiruan (JST) Backpropagation**.
""")

# Sidebar untuk navigasi
st.sidebar.header("Navigasi")
menu = st.sidebar.selectbox("Pilih Tampilan", ["Dataset & Tren", "Proses Training", "Evaluasi Model"])

# Load Data
@st.cache_data
def load_data():
    df_monthly = pd.read_csv('beras_medium_monthly.csv')
    df_results = pd.read_csv('prediction_results.csv')
    try:
        df_loss = pd.read_csv('loss_curve.csv')
    except:
        df_loss = None
    return df_monthly, df_results, df_loss

try:
    df_monthly, df_results, df_loss = load_data()

    if menu == "Dataset & Tren":
        st.header("📈 Tren Harga Historis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Sampel Data Terolah")
            st.write(df_monthly.tail(10))
            st.info("Data ini adalah rata-rata harga bulanan seluruh kabupaten di Jawa Timur.")

        with col2:
            st.subheader("Grafik Fluktuasi Harga")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(pd.to_datetime(df_monthly['periode_update']), df_monthly['jumlah'], marker='o', color='green')
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Harga (Rp)")
            st.pyplot(fig)

    elif menu == "Proses Training":
        st.header("🧠 Proses Pembelajaran JST")
        st.markdown("""
        **Arsitektur Model:**
        - **Input:** 2 Fitur (Lag-1, Lag-2)
        - **Hidden Layers:** 2 Lapis (12, 8 Neuron)
        - **Optimizer:** Adam
        - **Activation:** ReLU
        """)
        
        st.subheader("Analisis Loss Curve")
        if df_loss is not None:
            fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
            ax_loss.plot(df_loss['loss'], color='red')
            ax_loss.set_title("Penurunan Error per Epoch")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            st.pyplot(fig_loss)
            st.success("Model berhasil konvergen (Error menurun dan melandai).")
        else:
            st.warning("Data Loss Curve belum ditemukan. Jalankan model_training.ipynb terlebih dahulu.")

    elif menu == "Evaluasi Model":
        st.header("📊 Evaluasi Kinerja & Akurasi")
        
        # Metrik
        actual = df_results['actual']
        pred = df_results['prediction']
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred)**2))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"Rp {mae:.2f}")
        col2.metric("RMSE", f"Rp {rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        col4.metric("Total Data Uji", f"{len(df_results)} Bulan")

        st.divider()

        # Grafik Perbandingan
        st.subheader("Perbandingan Harga Aktual vs Prediksi")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(actual, label='Harga Aktual', marker='o', linestyle='-', color='blue')
        ax2.plot(pred, label='Harga Prediksi JST', marker='x', linestyle='--', color='orange')
        ax2.set_ylabel("Harga (Rp)")
        ax2.legend()
        st.pyplot(fig2)

        # Scatter Plot
        st.subheader("Korelasi Prediksi")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.regplot(x=actual, y=pred, ax=ax3, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        ax3.set_xlabel("Harga Aktual (Rp)")
        ax3.set_ylabel("Harga Prediksi (Rp)")
        st.pyplot(fig3)

except FileNotFoundError:
    st.error("File data tidak ditemukan. Pastikan Anda sudah menjalankan seluruh notebook (Preprocess & Training) terlebih dahulu.")

st.sidebar.markdown("---")
st.sidebar.write("Dibuat untuk Final Project AI 2026")
