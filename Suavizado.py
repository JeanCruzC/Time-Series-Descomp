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
file = st.file_uploader('Sube un archivo CSV o Excel con tu serie temporal', type=['csv','xlsx'])
if not file:
    st.info('Por favor, sube tu archivo para iniciar el análisis.')
    st.stop()

# Carga de datos
try:
    df = pd.read_csv(file)
except:
    df = pd.read_excel(file)
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
ts_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not ts_cols:
    st.error('No se encontró columna numérica de valores.')
    st.stop()
val_col = ts_cols[0]
st.success(f'Columna de valor detectada: **{val_col}**')

# Inferencia de frecuencia y ajuste
freq = pd.infer_freq(df.index) or df.index.inferred_freq or 'D'
df = df.asfreq(freq)
st.success(f'Frecuencia inferida: **{freq}**')

# Estimación de periodo estacional
freq_letter = ''.join(filter(str.isalpha, freq))
default_period = {'H':24, 'D':7, 'T':60, 'S':60, 'W':52}
period = default_period.get(freq_letter, 7)
st.success(f'Periodo estacional estimado: **{period}**')

# Descomposición
decomp = seasonal_decompose(df[val_col], model='additive', period=period)
st.subheader('🔍 Descomposición de la serie')
fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
axes[0].plot(decomp.observed); axes[0].set_title('Observado')
axes[1].plot(decomp.trend);    axes[1].set_title('Tendencia')
axes[2].plot(decomp.seasonal); axes[2].set_title('Estacionalidad')
axes[3].plot(decomp.resid);    axes[3].set_title('Residuo')
plt.tight_layout()
st.pyplot(fig)

# Suavizado automático (media móvil)
df['Smoothed'] = df[val_col].rolling(window=period, center=True, min_periods=1).mean()
st.subheader('📈 Original vs Smoothed')
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(df[val_col], label='Original')
ax2.plot(df['Smoothed'], label='Smoothed', linewidth=2)
ax2.set_title('Serie vs Suavizado')
ax2.legend()
st.pyplot(fig2)

# Detección de outliers y limpieza
df['z_score'] = zscore(df[val_col].fillna(method='ffill'))
threshold = 3
df_clean = df[np.abs(df['z_score']) < threshold]
st.subheader('🚫 Serie sin Outliers')
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(df_clean[val_col], label='Cleaned')
ax3.set_title('Outliers Eliminados')
st.pyplot(fig3)

# Resumen ejecutivo
max_val = df[val_col].max()
max_time = df[val_col].idxmax()
avg_val = df[val_col].mean()
std_val = df[val_col].std()
st.subheader('📝 Resumen Ejecutivo')
st.markdown(f"- **Máximo**: {max_val:.2f} en {max_time}")
st.markdown(f"- **Promedio**: {avg_val:.2f}")
st.markdown(f"- **Desviación estándar**: {std_val:.2f}")

# Alertas de picos
thresh_peak = avg_val + 2*std_val
peaks = df[df[val_col] > thresh_peak]
st.subheader('⚠️ Alertas de Picos')
if peaks.empty:
    st.write('No se detectaron picos críticos.')
else:
    st.write(f'Se detectaron {len(peaks)} picos (>{thresh_peak:.2f}):')
    st.dataframe(peaks[[val_col]])
    # Gráfico de picos señalados
    fig4, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(df[val_col], label='Original')
    ax4.scatter(peaks.index, peaks[val_col], color='red', label='Picos')
    ax4.set_title('Picos detectados en la serie')
    ax4.legend()
    st.pyplot(fig4)

# Exportar
export_df = df_clean.copy()
export_df['Smoothed'] = df['Smoothed']
export_df['Peak'] = df[val_col] > thresh_peak
st.subheader('💾 Exportar Resultados')
csv = export_df.reset_index()[[date_col, val_col, 'Smoothed', 'Peak']].to_csv(index=False).encode('utf-8')
st.download_button('Descargar CSV con Peaks', csv, 'serie_con_peaks.csv', 'text/csv')
