import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Configuración de la app
st.set_page_config(page_title='Análisis Automático de Series Temporales', layout='wide')
st.title('📊 Análisis Automático de Series Temporales')

# Subida de datos
data_file = st.file_uploader('Sube un archivo CSV o Excel con tu serie temporal', type=['csv','xlsx'])
if not data_file:
    st.info('Por favor, sube tu archivo para iniciar el análisis.')
    st.stop()

# Carga de datos
try:
    df = pd.read_csv(data_file)
except:
    df = pd.read_excel(data_file)
st.write('### Datos originales')
st.dataframe(df.head())

# Detección automática de columna de fecha
date_col = None
for col in df.columns:
    try:
        df[col] = pd.to_datetime(df[col])
        date_col = col
        break
    except:
        continue
if date_col is None:
    st.error('No se encontró columna de fecha.')
    st.stop()
st.success(f'Columna de fecha detectada: **{date_col}**')
df = df.set_index(date_col).sort_index()

# Detección automática de columna de valor
time_series_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not time_series_cols:
    st.error('No se encontró columna numérica de valores.')
    st.stop()
value_col = time_series_cols[0]
st.success(f'Columna de valor detectada: **{value_col}**')

# Inferencia de frecuencia y asignación
freq = pd.infer_freq(df.index) or df.index.inferred_freq or 'D'
df = df.asfreq(freq)
st.success(f'Frecuencia inferida: **{freq}**')

# Estimación de periodo estacional
freq_letter = ''.join(filter(str.isalpha, freq))
default_periods = {'H':24, 'D':7, 'T':60, 'S':60, 'W':52}
period = default_periods.get(freq_letter, 7)
st.success(f'Periodo estacional estimado: **{period}**')

# Descomposición de la serie
st.subheader('🔍 Descomposición de la serie')
decomp = seasonal_decompose(df[value_col], model='additive', period=period)
fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
axes[0].plot(decomp.observed); axes[0].set_title('Observado')
axes[1].plot(decomp.trend);    axes[1].set_title('Tendencia')
axes[2].plot(decomp.seasonal); axes[2].set_title('Estacionalidad')
axes[3].plot(decomp.resid);    axes[3].set_title('Residuo')
plt.tight_layout()
st.pyplot(fig)

# Suavizado automático (media móvil)
st.subheader('📈 Suavizado automático')
df['Smoothed'] = df[value_col].rolling(window=period, center=True, min_periods=1).mean()
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(df[value_col], label='Original')
ax2.plot(df['Smoothed'], label='Smoothed', linewidth=2)
ax2.set_title('Original vs Smoothed')
ax2.legend()
st.pyplot(fig2)

# Detección y eliminación de outliers
st.subheader('🚫 Eliminación de outliers')
df['z_score'] = zscore(df[value_col].fillna(method='ffill'))
threshold = 3
df_clean = df[np.abs(df['z_score']) < threshold]
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(df_clean[value_col], label='Cleaned')
ax3.set_title('Serie sin outliers (|z|<3)')
st.pyplot(fig3)

# Resumen Ejecutivo y Alertas de Picos
st.subheader('📝 Resumen Ejecutivo')
max_val = df[value_col].max()
max_time = df[value_col].idxmax()
avg_val = df[value_col].mean()
std_val = df[value_col].std()
st.markdown(f"- **Máximo**: {max_val:.2f} en {max_time}")
st.markdown(f"- **Promedio**: {avg_val:.2f}")
st.markdown(f"- **Desviación estándar**: {std_val:.2f}")

# Alertas de Picos: valores > promedio + 2*std
thresh_peak = avg_val + 2 * std_val
peaks = df[df[value_col] > thresh_peak][[value_col]]
st.subheader('⚠️ Alertas de Picos')
if peaks.empty:
    st.write('No se detectaron picos críticos.')
else:
    st.write(f'Se detectaron {len(peaks)} picos por encima de {thresh_peak:.2f}:')
    st.dataframe(peaks)

# Descargar resultados
st.subheader('💾 Descargar Datos Procesados')
output = df_clean.reset_index()[[date_col, value_col, 'Smoothed']]
csv = output.to_csv(index=False).encode('utf-8')
st.download_button('Descargar CSV', csv, 'serie_procesada.csv', 'text/csv')
