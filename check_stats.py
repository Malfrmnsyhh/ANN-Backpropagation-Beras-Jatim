import pandas as pd

df = pd.read_csv('beras_medium_monthly.csv')
print('=== Statistik Data Asli ===')
print(df[['jumlah','lag_1','lag_2']].describe())
print()

df2 = pd.read_csv('beras_scaled.csv')
print('=== Statistik Data Scaled ===')
print(df2.describe())
print()

print('=== Min/Max asli ===')
print(f'Min jumlah: {df["jumlah"].min():.2f}')
print(f'Max jumlah: {df["jumlah"].max():.2f}')
print(f'Min lag_1: {df["lag_1"].min():.2f}')
print(f'Max lag_1: {df["lag_1"].max():.2f}')
print(f'Min lag_2: {df["lag_2"].min():.2f}')
print(f'Max lag_2: {df["lag_2"].max():.2f}')
print()

print('=== 5 Data Pertama Asli ===')
print(df[['periode_update','jumlah','lag_1','lag_2']].head().to_string(index=False))
print()

print('=== 5 Data Pertama Scaled ===')
print(df2.head().to_string(index=False))
print()

print(f'Jumlah total data (sampel): {len(df)}')
print(f'Data training (80%): {int(len(df)*0.8)}')
print(f'Data testing (20%): {len(df) - int(len(df)*0.8)}')
