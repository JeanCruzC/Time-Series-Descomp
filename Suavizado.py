import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
import matplotlib.pyplot as plt

st.set_page_config(page_title='Análisis Automático de Series Temporales', layout='wide')
st.title('📊 Análisis Automático de Series Temporales')

# Subida de datos
dato = st.file_uploader('Sube un archivo CSV o Excel con tu serie temporal', type=['csv','xlsx'])
if dato:
    # Carga de datos
    try:
        df = pd.read_csv(dato)
    except:
        df = pd.read_excel(dato)
    st.write('### Datos originales')
    st.dataframe(df.head())

    # Detección automática de columna de fecha
    fecha_cols = df.select_dtypes(include=['datetime','datetimetz']).columns.tolist()
    if not fecha_cols:
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                fecha_cols = [col]
                break
            except:
                continue
    fecha = fecha_cols[0]
    st.success(f'Columna de fecha detectada: **{fecha}**')
    df[fecha] = pd.to_datetime(df[fecha])
    df = df.set_index(fecha).sort_index()

    # Detección automática de columna de tráfico
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    trafico = num_cols[0]
    st.success(f'Columna de tráfico detectada: **{trafico}**')

    # Inferir frecuencia y aplicar
    freq = pd.infer_freq(df.index)
    if not freq:
        freq = df.index.inferred_freq or 'D'
    df = df.asfreq(freq)
    st.success(f'Frecuencia de la serie inferida: **{freq}**')

    # Determinar periodo estacional según frecuencia
    mapa_periodo = {'H':24, 'D':7, 'T':60, 'S':60, 'W':52}
    clave = ''.join(filter(str.isalpha, freq))
    periodo = mapa_periodo.get(clave, 7)
    st.success(f'Periodo estacional estimado: **{periodo}**')

    # Modelo de descomposición por defecto
    modelo = 'additive'
    st.success(f'Modelo de descomposición: **{modelo}**')

    # Descomposición
    st.subheader('🔍 Descomposición de la serie')
    descomp = seasonal_decompose(df[trafico], model=modelo, period=periodo)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(descomp.observed); axes[0].set_title('Observado')
    axes[1].plot(descomp.trend);    axes[1].set_title('Tendencia')
    axes[2].plot(descomp.seasonal); axes[2].set_title('Estacionalidad')
    axes[3].plot(descomp.resid);    axes[3].set_title('Residuo')
    plt.tight_layout()
    st.pyplot(fig)

    # Suavizado automático (media móvil)
    df['Suavizado'] = df[trafico].rolling(window=periodo, center=True).mean()
    st.subheader('📈 Serie original vs suavizada')
    comparacion = df[[trafico, 'Suavizado']].dropna().rename(columns={trafico: 'Original'})
    st.line_chart(comparacion)
    st.write('Muestra de datos comparados:')
    st.dataframe(comparacion.head())

    # Eliminación de valores atípicos (z-score < 3)
    df['z_score'] = zscore(df[trafico].fillna(method='ffill'))
    limpio = df[np.abs(df['z_score']) < 3]
    st.subheader('🚫 Serie tras eliminar outliers')
    st.line_chart(limpio[trafico])

    # Descarga de la serie limpia
    datos_csv = limpio.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button('Descargar serie limpia (CSV)', datos_csv, 'serie_limpia.csv', 'text/csv')
else:
    st.info('Por favor, sube tu archivo para iniciar el análisis.')
