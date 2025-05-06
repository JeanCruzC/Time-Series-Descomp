import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
import matplotlib.pyplot as plt

st.set_page_config(page_title='Análisis de Serie Temporal', layout='wide')
st.title('📈 Descomposición y Suavizado de Series Temporales')

# Carga de archivo
uploaded_file = st.file_uploader('Sube un archivo CSV o Excel con columnas de fecha y tráfico', type=['csv','xlsx'])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_excel(uploaded_file)
    st.write('### Vista previa de los datos')
    st.dataframe(df.head())

    # Selección de columnas
    date_col = st.selectbox('Selecciona la columna de fecha', df.columns)
    traffic_col = st.selectbox('Selecciona la columna de tráfico', df.columns)

    # Preparar la serie temporal
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Definir frecuencia y periodo estacional
    freq = st.selectbox('Frecuencia de la serie', ['D', 'H', 'T', 'W'], index=1, help='D=Diario, H=Horario, T=Minuto, W=Semanal')
    df = df.asfreq(freq)
    period = st.number_input('Periodo estacional (p.ej. 24 para datos horarios)', min_value=1, value=24)
    model = st.selectbox('Modelo de descomposición', ['additive', 'multiplicative'])

    # Descomposición
    st.subheader('🧩 Descomposición de la Serie')
    decomposition = seasonal_decompose(df[traffic_col], model=model, period=period)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(decomposition.observed); axes[0].set_title('Observado')
    axes[1].plot(decomposition.trend); axes[1].set_title('Tendencia')
    axes[2].plot(decomposition.seasonal); axes[2].set_title('Estacionalidad')
    axes[3].plot(decomposition.resid); axes[3].set_title('Residuo')
    plt.tight_layout()
    st.pyplot(fig)

    # Suavizado con Media Móvil
    st.subheader('🛠️ Suavizado con Media Móvil')
    window = st.slider('Tamaño de ventana para suavizado', min_value=2, max_value=period * 2, value=period)
    df['Suavizado'] = df[traffic_col].rolling(window=window, center=True).mean()

    # Comparación Serie Original vs Suavizada
    st.write('### Comparación: Serie Original vs Suavizada')
    df_comp = df[[traffic_col, 'Suavizado']].dropna().rename(columns={traffic_col: 'Original'})
    st.line_chart(df_comp)
    st.write('#### Muestra de datos comparados')
    st.dataframe(df_comp.head(10))

    # Detección y eliminación de outliers
    st.subheader('❗ Eliminación de Valores Atípicos')
    threshold = st.slider('Umbral de z-score', min_value=0.0, max_value=5.0, value=3.0)
    df['z_score'] = zscore(df[traffic_col].fillna(method='ffill'))
    df_clean = df[np.abs(df['z_score']) < threshold]
    st.write('### Serie Tras Eliminación de Outliers')
    st.line_chart(df_clean[traffic_col].rename('Limpia'))

    # Descarga de datos procesados
    st.subheader('💾 Descargar Datos Procesados')
    csv = df_clean.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button('Descargar serie limpia (CSV)', data=csv, file_name='serie_limpia.csv', mime='text/csv')
