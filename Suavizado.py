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
else:
    # Carga de datos
    try:
        df = pd.read_csv(data_file)
    except:
        df = pd.read_excel(data_file)
    st.write('### Datos originales')
    st.dataframe(df.head())

    # Detección automática de columna de fecha
    fecha_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col])
            df[col] = parsed
            fecha_col = col
            break
        except:
            continue
    if fecha_col is None:
        st.error('No se encontró columna de fecha.')
        st.stop()
    st.success(f'Columna de fecha detectada: **{fecha_col}**')
    df = df.set_index(fecha_col).sort_index()

    # Detección automática de columna numérica de valor
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error('No se encontró columna numérica de valores.')
        st.stop()
    valor_col = num_cols[0]
    st.success(f'Columna de valor detectada: **{valor_col}**')

    # Inferencia de frecuencia
    freq = pd.infer_freq(df.index) or df.index.inferred_freq or 'D'
    df = df.asfreq(freq)
    st.success(f'Frecuencia inferida: **{freq}**')

    # Estimación automática de periodo estacional
    freq_letter = ''.join(filter(str.isalpha, freq))
    default_periods = {'H':24, 'D':7, 'T':60, 'S':60, 'W':52}
    period = default_periods.get(freq_letter, 7)
    st.success(f'Periodo estacional estimado: **{period}**')

    # Descomposición de la serie
    st.subheader('🔍 Descomposición de la serie')
    decomposition = seasonal_decompose(df[valor_col], model='additive', period=period)
    fig, axes = plt.subplots(4,1,figsize=(10,8), sharex=True)
    axes[0].plot(decomposition.observed); axes[0].set_title('Observado')
    axes[1].plot(decomposition.trend);    axes[1].set_title('Tendencia')
    axes[2].plot(decomposition.seasonal); axes[2].set_title('Estacionalidad')
    axes[3].plot(decomposition.resid);    axes[3].set_title('Residuo')
    plt.tight_layout()
    st.pyplot(fig)

    # Suavizado automático con media móvil (ventana = periodo)
    st.subheader('📈 Suavizado automático')
    df['Smoothed'] = df[valor_col].rolling(window=period, center=True, min_periods=1).mean()
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(df[valor_col], label='Original')
    ax2.plot(df['Smoothed'], label='Smoothed', linewidth=2)
    ax2.set_title('Original vs Smoothed')
    ax2.legend()
    st.pyplot(fig2)

    # Detección y eliminación de outliers (z-score)
    st.subheader('🚫 Valores atípicos eliminados')
    df['z_score'] = zscore(df[valor_col].fillna(method='ffill'))
    threshold = 3
    df_clean = df[np.abs(df['z_score']) < threshold]
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(df_clean[valor_col], label='Cleaned')
    ax3.set_title('Serie sin outliers (|z|<3)')
    st.pyplot(fig3)

    # Descargar resultados
    st.subheader('💾 Descargar datos procesados')
    result = df_clean.reset_index()[[fecha_col, valor_col, 'Smoothed']]
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar CSV', csv, 'serie_procesada.csv', 'text/csv')
