import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Final Project AI", layout="wide", page_icon="🎓")

# Custom CSS untuk mempercantik UI
st.markdown("""
<style>
    /* Styling untuk Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--background-color);
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #2e7bcf;
    }
    
    /* Menyembunyikan elemen bawaan Streamlit agar terlihat lebih profesional */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Styling header */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Dashboard Prediksi Harga Beras Medium - Jawa Timur")
st.markdown("""
**Prediksi Harga Pangan Menggunakan Jaringan Saraf Tiruan (JST) Backpropagation**
""")

# Sidebar untuk navigasi
with st.sidebar:
    st.header("Navigasi Utama")
    menu = st.radio("Pilih Tampilan:", [
        "1. Overview & Dataset", 
        "2. Preprocessing Data", 
        "3. Arsitektur & Training JST", 
        "4. Evaluasi Kinerja",
        "5. Prediksi Interaktif"
    ])
    st.markdown("---")

# Load Data & Models
@st.cache_data
def load_data():
    try:
        df_monthly = pd.read_csv('beras_medium_monthly.csv')
        df_monthly['periode_update'] = pd.to_datetime(df_monthly['periode_update'])
        df_results = pd.read_csv('prediction_results.csv')
        df_scaled = pd.read_csv('beras_scaled.csv')
    except Exception as e:
        return None, None, None, None

    try:
        df_loss = pd.read_csv('loss_curve.csv')
    except:
        df_loss = None
        
    return df_monthly, df_results, df_loss, df_scaled

@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

df_monthly, df_results, df_loss, df_scaled = load_data()
model, scaler = load_models()

if df_monthly is None or df_results is None:
    st.error("⚠️ File data tidak ditemukan. Pastikan Anda sudah menjalankan seluruh proses preprocessing & training.")
else:
    if menu == "1. Overview & Dataset":
        st.header("Overview Penelitian")
        st.markdown("""
        Penelitian ini bertujuan memprediksi fluktuasi harga beras medium di tingkat kabupaten/kota se-Jawa Timur 
        menggunakan data historis *time series*. Algoritma JST Backpropagation digunakan sebagai "otak" untuk mengenali pola kenaikan atau penurunan harga.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", f"{len(df_monthly)} Bulan")
        col2.metric("Proporsi Train/Test", "80% / 20%")
        col3.metric("Arsitektur JST", "2 - 12 - 8 - 1")
        loss_val = df_loss['loss'].iloc[-1] if df_loss is not None else 0
        col4.metric("Final Loss", f"{loss_val:.4f}")
        
        st.divider()
        
        st.header("Visualisasi Tren Harga Beras")
        st.write("Grafik pergerakan rata-rata harga beras medium di Jawa Timur.")
        
        # Interaktif Line Chart
        chart_data = df_monthly.set_index('periode_update')[['jumlah']]
        st.line_chart(chart_data, color="#2e7bcf", height=400)
        
        # Insight
        max_idx = df_monthly['jumlah'].idxmax()
        max_price = df_monthly.loc[max_idx, 'jumlah']
        max_date = df_monthly.loc[max_idx, 'periode_update'].strftime('%B %Y')
        st.info(f"**Insight:** Berdasarkan data di atas, harga beras mencapai titik tertingginya sebesar **Rp {max_price:,.0f}** pada **{max_date}**.")

    elif menu == "2. Preprocessing Data":
        st.header("Pra-Pemrosesan Data (Preprocessing)")
        st.markdown("""
        Sebelum dimasukkan ke dalam model JST, data harga mentah diagregasi secara bulanan dan diubah menjadi format *supervised learning* menggunakan teknik **Sliding Window**.
        Lalu, data dinormalisasi menggunakan **Min-Max Scaling** agar nilainya berada di rentang 0 hingga 1. Hal ini wajib dilakukan agar fungsi aktivasi JST tidak mengalami saturasi.
        """)
        
        st.subheader("1. Sliding Window (Feature Engineering)")
        st.write("Kita menggunakan 2 bulan sebelumnya (Lag-1 dan Lag-2) untuk memprediksi harga bulan ini (Target).")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Aktual (Rupiah)**")
            st.dataframe(df_monthly[['periode_update', 'lag_2', 'lag_1', 'jumlah']].tail(5), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Data Terskalakan (0 - 1)**")
            if df_scaled is not None:
                st.dataframe(df_scaled[['lag_2_scaled', 'lag_1_scaled', 'jumlah_scaled']].tail(5), hide_index=True, use_container_width=True)
            else:
                st.warning("Data beras_scaled.csv tidak ditemukan.")
                
        st.success("Data yang terskalakan inilah yang digunakan untuk melatih Jaringan Saraf Tiruan.")

    elif menu == "3. Arsitektur & Training JST":
        st.header("Arsitektur & Pelatihan Jaringan Saraf Tiruan")
        
        col_arch, col_train = st.columns([1, 1.5])
        
        with col_arch:
            st.subheader("Arsitektur Model")
            st.markdown("""
            **Struktur Layer:**
            - **Input Layer:** 2 Neuron (Lag-1 & Lag-2)
            - **Hidden Layer 1:** 12 Neuron
            - **Hidden Layer 2:** 8 Neuron
            - **Output Layer:** 1 Neuron (Prediksi Harga)
            """)
            st.markdown("""
            **Hyperparameters:**
            - Optimizer: `Adam`
            - Learning Rate: `0.001`
            - Activation: `ReLU`
            - Max Epochs: `3000`
            """)
            
        with col_train:
            st.subheader("Analisis Konvergensi (Loss Curve)")
            if df_loss is not None:
                st.line_chart(df_loss['loss'], color="#d62728", height=300)
                
                final_epoch = len(df_loss)
                final_loss = df_loss['loss'].iloc[-1]
                st.success(f"✔ **Status:** Model telah mencapai konvergensi stabil pada Epoch ke-{final_epoch} dengan Final Loss **{final_loss:.5f}**.")
            else:
                st.warning("Data Loss Curve belum ditemukan.")

    elif menu == "4. Evaluasi Kinerja":
        st.header("Evaluasi Kinerja & Akurasi Prediksi")
        
        actual = df_results['actual']
        pred = df_results['prediction']
        
        mae = np.mean(np.abs(actual - pred))
        mse = np.mean((actual - pred)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        st.markdown("Pengujian dilakukan pada 20% data terakhir (Data Testing) yang belum pernah dilihat oleh model.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE (Rata-rata Error)", f"Rp {mae:.2f}")
        col2.metric("RMSE", f"Rp {rmse:.2f}")
        col3.metric("MSE", f"{mse:.0f}")
        col4.metric("MAPE", f"{mape:.2f}%")
        
        if mape < 10:
            st.success("**Insight:** Nilai MAPE berada di bawah 10% sehingga model tergolong **Sangat Akurat** untuk peramalan harga beras.")
        
        st.divider()

        st.subheader("Perbandingan Aktual vs Prediksi JST")
        eval_df = pd.DataFrame({
            'Harga Aktual': actual,
            'Prediksi JST': pred
        })
        st.line_chart(eval_df, color=["#1f77b4", "#ff7f0e"], height=350)
        
        col_scat, col_res = st.columns(2)
        
        with col_scat:
            st.subheader("Scatter Plot Aktual vs Prediksi")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x=actual, y=pred, ax=ax, scatter_kws={'alpha':0.6, 'color':'#1f77b4'}, line_kws={'color':'#d62728', 'linewidth':2})
            ax.set_xlabel("Harga Aktual (Rp)")
            ax.set_ylabel("Harga Prediksi (Rp)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.caption("Titik yang dekat dengan garis lurus menunjukkan korelasi prediksi yang kuat.")
            
        with col_res:
            st.subheader("Analisis Error (Distribusi Residual)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            residuals = actual - pred
            sns.histplot(residuals, kde=True, color="purple", ax=ax2)
            ax2.set_xlabel("Residual (Aktual - Prediksi)")
            ax2.set_ylabel("Frekuensi")
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
            st.caption("Distribusi error yang berpusat di sekitar angka 0 menunjukkan model tidak memiliki bias prediksi yang parah.")

    elif menu == "5. Prediksi Interaktif":
        st.header("Simulasi Prediksi Harga Beras")
        st.markdown("Masukkan rata-rata harga beras Jawa Timur pada dua bulan terakhir untuk memprediksi harga bulan depan.")
        
        if model is None or scaler is None:
            st.error("Model JST atau Scaler belum diload. Pastikan save_models.py telah dijalankan untuk men-generate model.pkl dan scaler.pkl.")
        else:
            with st.form("predict_form"):
                col1, col2 = st.columns(2)
                with col1:
                    lag_1 = st.number_input("Harga 1 Bulan Lalu (Lag-1) - Rp", min_value=5000.0, max_value=30000.0, value=12000.0, step=100.0)
                with col2:
                    lag_2 = st.number_input("Harga 2 Bulan Lalu (Lag-2) - Rp", min_value=5000.0, max_value=30000.0, value=12000.0, step=100.0)
                
                submit_button = st.form_submit_button(label="Prediksi Harga")
                
            if submit_button:
                # Format fitur untuk scaler: [jumlah_dummy, lag_1, lag_2]
                input_data = np.array([[0, lag_1, lag_2]])
                scaled_input = scaler.transform(input_data)
                
                # Model dilatih hanya dengan 2 fitur lag
                X_pred = np.array([[scaled_input[0, 1], scaled_input[0, 2]]])
                
                # Prediksi
                pred_scaled = model.predict(X_pred)
                
                # Inverse Transform: [pred_scaled, 0, 0]
                inverse_input = np.array([[pred_scaled[0], 0, 0]])
                pred_actual = scaler.inverse_transform(inverse_input)[0, 0]
                
                st.subheader("Hasil Prediksi Bulan Berikutnya")
                
                delta_val = pred_actual - lag_1
                if delta_val > 0:
                    delta_str = f"Naik Rp {delta_val:,.0f}"
                else:
                    delta_str = f"Turun Rp {abs(delta_val):,.0f}"
                    
                st.metric("Prediksi Harga JST", f"Rp {pred_actual:,.0f}", delta=delta_str, delta_color="inverse")
                
                if delta_val > 500:
                    st.warning("Peringatan: Terdapat indikasi kenaikan harga yang cukup signifikan bulan depan.")
                elif delta_val < -500:
                    st.success("Kabar Baik: Harga beras diprediksi akan mengalami penurunan tajam bulan depan.")
                else:
                    st.info("Harga beras diprediksi relatif stabil.")
