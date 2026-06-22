import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Prediksi Harga Beras Jatim", page_icon="🌾", layout="wide")

# Custom CSS untuk mempercantik UI
st.markdown("""
<style>
    /* Styling untuk Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #f0f2f6;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2e7bcf;
    }
    
    /* Menyembunyikan elemen bawaan Streamlit agar terlihat lebih profesional */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Styling header */
    h1, h2, h3 {
        color: #1f3a52;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Dashboard Prediksi Harga Beras Medium - Jawa Timur")
st.markdown("""
Selamat datang di **Sistem Cerdas Prediksi Harga Pangan**. Dashboard ini menampilkan hasil analisis fluktuasi harga beras medium di wilayah Jawa Timur menggunakan metode **Jaringan Saraf Tiruan (JST) Backpropagation**.
""")

# Sidebar untuk navigasi
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100) # Ilustrasi beras/pangan
    st.header("Navigasi Utama")
    menu = st.radio("Pilih Tampilan:", ["Dataset & Tren", "Proses Training", "Evaluasi Model"])
    st.markdown("---")
    st.write("✨ **Dibuat untuk Final Project AI 2026**")

# Load Data
@st.cache_data
def load_data():
    try:
        df_monthly = pd.read_csv('beras_medium_monthly.csv')
        df_monthly['periode_update'] = pd.to_datetime(df_monthly['periode_update'])
        df_results = pd.read_csv('prediction_results.csv')
    except Exception as e:
        return None, None, None

    try:
        df_loss = pd.read_csv('loss_curve.csv')
    except:
        df_loss = None
        
    return df_monthly, df_results, df_loss

df_monthly, df_results, df_loss = load_data()

if df_monthly is None or df_results is None:
    st.error("⚠️ File data tidak ditemukan. Pastikan Anda sudah menjalankan seluruh notebook (Preprocess & Training) terlebih dahulu.")
else:
    if menu == "Dataset & Tren":
        st.header("📈 Tren Harga Historis")
        st.write("Analisis pergerakan rata-rata harga beras medium tingkat kabupaten/kota di Jawa Timur.")
        
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.subheader("Sampel Data Terolah")
            st.dataframe(df_monthly[['periode_update', 'jumlah']].tail(10), use_container_width=True, hide_index=True)
            st.info("💡 Data ini adalah hasil agregasi rata-rata harga bulanan.")

        with col2:
            st.subheader("Grafik Fluktuasi Harga")
            # Menggunakan line chart native streamlit yang interaktif
            chart_data = df_monthly.set_index('periode_update')[['jumlah']]
            st.line_chart(chart_data, color="#2e7bcf", height=400)

    elif menu == "Proses Training":
        st.header("🧠 Proses Pembelajaran Model JST")
        
        col_info, col_chart = st.columns([1, 2])
        
        with col_info:
            st.markdown("""
            ### Arsitektur Model:
            - **Input Layer:** 2 Fitur (Lag-1, Lag-2)
            - **Hidden Layers:** 2 Lapis (12 & 8 Neuron)
            - **Output Layer:** 1 Neuron (Prediksi Harga)
            - **Optimizer:** Adam
            - **Activation:** ReLU / Tanh
            """)
            st.success("Tujuan fase ini adalah menurunkan *Error* / *Loss* serendah mungkin sehingga model bisa mengenali pola fluktuasi harga.")
            
        with col_chart:
            st.subheader("Analisis Loss Curve (Penurunan Error)")
            if df_loss is not None:
                # Interaktif loss curve
                st.line_chart(df_loss['loss'], color="#d62728", height=350)
                st.caption("Grafik menunjukkan bagaimana error semakin mengecil seiring bertambahnya epoch pembelajaran.")
            else:
                st.warning("Data Loss Curve belum ditemukan. Jalankan model_training.ipynb terlebih dahulu.")

    elif menu == "Evaluasi Model":
        st.header("📊 Evaluasi Kinerja & Akurasi Prediksi")
        
        # Perhitungan Metrik sesuai standar formula
        actual = df_results['actual']
        pred = df_results['prediction']
        
        mae = np.mean(np.abs(actual - pred))
        mse = np.mean((actual - pred)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        st.markdown("Berikut adalah hasil perhitungan metrik evaluasi (Error Rate). Semakin kecil nilai error, semakin akurat model yang dibuat.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE (Rata-rata Error)", f"Rp {mae:.2f}")
        col2.metric("RMSE", f"Rp {rmse:.2f}")
        col3.metric("MSE", f"{mse:.0f}")
        col4.metric("MAPE", f"{mape:.2f}%")
        col5.metric("Total Data Uji", f"{len(df_results)} Bulan")

        st.divider()

        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.subheader("Perbandingan Aktual vs Prediksi JST")
            # Menyiapkan data untuk chart interaktif
            eval_df = pd.DataFrame({
                'Harga Aktual': actual,
                'Prediksi JST': pred
            })
            st.line_chart(eval_df, color=["#1f77b4", "#ff7f0e"], height=400)
            
        with col_chart2:
            st.subheader("Korelasi Prediksi")
            # Menggunakan matplotlib & seaborn untuk scatter plot
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.regplot(x=actual, y=pred, ax=ax, scatter_kws={'alpha':0.6, 'color':'#1f77b4'}, line_kws={'color':'#d62728', 'linewidth':2})
            ax.set_xlabel("Harga Aktual (Rp)", fontsize=10)
            ax.set_ylabel("Harga Prediksi (Rp)", fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.caption("Jika titik berkumpul di sekitar garis merah, korelasi prediksi sangat baik.")
