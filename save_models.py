


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

# 1. Load Data Mentah
df = pd.read_csv('beras_medium_monthly.csv')

# Preprocessing: Sliding Window
df['lag_1'] = df['jumlah'].shift(1)
df['lag_2'] = df['jumlah'].shift(2)
df.dropna(inplace=True)

# 2. Inisialisasi & Fit Scaler
scaler = MinMaxScaler()
features_to_scale = ['jumlah', 'lag_1', 'lag_2']
df_scaled_values = scaler.fit_transform(df[features_to_scale])

df_scaled = pd.DataFrame(df_scaled_values, columns=['jumlah_scaled', 'lag_1_scaled', 'lag_2_scaled'])

# 3. Menentukan Fitur (X) dan Target (y)
X_scaled = df_scaled[['lag_1_scaled', 'lag_2_scaled']].values
y_scaled = df_scaled['jumlah_scaled'].values.reshape(-1, 1)

# 4. Split Data (Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)

# 5. Inisialisasi Model JST Backpropagation
model = MLPRegressor(
    hidden_layer_sizes=(12, 8), 
    activation='relu', 
    solver='adam', 
    max_iter=3000, 
    learning_rate_init=0.001, 
    learning_rate='adaptive',
    tol=1e-6,
    n_iter_no_change=200,
    early_stopping=False,
    random_state=42,
    verbose=False
)

# 6. Proses Training
model.fit(X_train, y_train.ravel())

# 7. Simpan Model & Scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')

print("Model dan Scaler berhasil disimpan sebagai model.pkl dan scaler.pkl")
