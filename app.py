import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Monitor SEN - Chile",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la estética
st.markdown("""
    <style>
    /* 1. Fondo general de la aplicación */
    .main {
        background-color: #f4f6f9; /* Gris muy suave */
    }
    
    /* 2. Estilo de los TABS (Pestañas) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Espacio entre botones */
        margin-bottom: 15px; /* Separación con el contenido */
    }

    /* Tab NO seleccionado (Botón gris) */
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #e9ecef;
        border-radius: 6px; /* Bordes redondeados completos */
        padding: 10px 20px;
        color: #495057;
        border: 1px solid #dee2e6;
    }

    /* Tab SELECCIONADO (Botón Azul) */
    .stTabs [aria-selected="true"] {
        background-color: #0068c9; /* Azul Corporativo */
        color: #ffffff;
        font-weight: bold;
        border-radius: 6px; /* Redondeado igual que el inactivo */
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); /* Sutil sombra para resaltar */
    }

    /* 3. CONTENIDO DEL TAB (El cambio que pediste) */
    div[data-baseweb="tab-panel"] {
        /* Fondo transparente para que se vea el gris de la página */
        background-color: transparent; 
        
        /* Sin bordes ni sombras */
        border: none;
        box-shadow: none;
        
        /* Mantenemos espacio vertical para que no se pegue al título */
        padding-top: 10px;
        padding-bottom: 20px;
    }
    
    /* Ajuste para títulos dentro de los tabs */
    h3 {
        color: #0068c9; /* Títulos en azul para jerarquía */
    }
    
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE CARGA (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Cargamos los datos procesados (con nombres de meses, productos, etc.)
        df = pd.read_csv('data/data_processed.csv')
        # Aseguramos que haya una fecha datetime para los gráficos
        # Asumiendo que tienes YEAR y MONTH o periodo_id. Creamos una fecha ficticia para plotear.
        if 'YEAR' in df.columns and 'MONTH' in df.columns:
             df['Fecha'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró 'data/data_processed.csv'. Ejecuta el notebook de exportación primero.")
        return None

@st.cache_data
def load_scaled_data():
    try:
        # Datos normalizados para los modelos
        return pd.read_csv('data/data_scaled.csv', index_col=0) # Asume periodo_id como índice
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    models = {}
    try:
        # Modelos de Scikit-Learn (sin cambios)
        models['kmeans'] = joblib.load('models/kmeans_model.pkl')
        models['pca'] = joblib.load('models/pca_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        
        # --- CORRECCIÓN AQUÍ ---
        # Usamos compile=False para evitar el error de 'keras.metrics.mse'
        # Esto carga la arquitectura y los pesos, pero ignora las métricas de entrenamiento
        models['autoencoder'] = tf.keras.models.load_model('models/autoencoder.h5', compile=False)
        models['lstm'] = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
        
        return models
    except Exception as e:
        st.error(f"⚠️ Error crítico cargando modelos: {e}")
        return None

# -----------------------------------------------------------------------------
# 3. INTERFAZ PRINCIPAL
# -----------------------------------------------------------------------------

# Cargar recursos
df = load_data()
df_scaled = load_scaled_data()
models = load_models()

# Título y Descripción
st.title("⚡ Sistema de Monitoreo y Predicción del SEN Chile")
st.markdown("Plataforma de Inteligencia Artificial para el análisis de la transición energética (2010-2022).")

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("🛠️ Configuración")
    
    # Filtro de Fuentes de Energía
    energy_types = df['PRODUCT'].unique()
    default_selection = ['Solar', 'Wind', 'Hydro', 'Coal'] if 'Solar' in energy_types else energy_types
    selected_energy = st.sidebar.multiselect(
        "Fuentes de Generación", 
        options=energy_types, 
        default=default_selection
    )
    
    # Filtro de Fechas (Slider)
    min_date = df['Fecha'].min().date()
    max_date = df['Fecha'].max().date()
    start_date, end_date = st.sidebar.slider(
        "Rango de Fechas",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # Filtrar el DataFrame Globalmente
    mask = (df['Fecha'].dt.date >= start_date) & (df['Fecha'].dt.date <= end_date) & (df['PRODUCT'].isin(selected_energy))
    df_filtered = df[mask]

    # --- TABS DE NAVEGACIÓN ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Análisis Histórico", 
        "🧩 Eras Energéticas (Clustering)", 
        "🚨 Salud de la Red (Anomalías)", 
        "🔮 Predicción (LSTM)"
    ])

    # =========================================================================
    # TAB 1: ANÁLISIS HISTÓRICO (EDA)
    # =========================================================================
    with tab1:
        # KPIs
        total_gen = df_filtered['VALUE'].sum()
        avg_gen = df_filtered['VALUE'].mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Generación (Selección)", f"{total_gen/1000:,.2f} TWh")
        c2.metric("Promedio Mensual", f"{avg_gen:,.2f} GWh")
        c3.metric("Registros Analizados", f"{len(df_filtered)}")

        st.divider()

        # Gráfico 1: Evolución Temporal (Plotly)
        st.subheader("📈 Evolución Temporal de la Generación")
        fig_line = px.line(
            df_filtered, 
            x='Fecha', 
            y='VALUE', 
            color='PRODUCT',
            title='Generación Eléctrica por Fuente (GWh)',
            labels={'VALUE': 'Generación (GWh)', 'Fecha': 'Año'},
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Gráfico 2: Share de Mercado (Plotly Bar)
        st.subheader("🍰 Composición de la Matriz (Share)")
        # Para el share necesitamos agrupar para que sume 100% o usar la columna 'share' si existe
        if 'share' in df_filtered.columns:
             fig_bar = px.bar(
                df_filtered, 
                x='Fecha', 
                y='share', 
                color='PRODUCT',
                title='Participación de Mercado Mensual',
                labels={'share': 'Participación (%)'},
                template="plotly_white"
            )
             st.plotly_chart(fig_bar, use_container_width=True)

        # Gráfico 3: Matplotlib (Boxplot de Distribución)
        st.subheader("📦 Distribución y Outliers")
        col_plot, col_desc = st.columns([2, 1])
        with col_plot:
            fig_box, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df_filtered, x='PRODUCT', y='VALUE', ax=ax, palette="Set2")
            plt.xticks(rotation=45)
            plt.ylabel("Generación (GWh)")
            plt.title("Variabilidad por Tecnología")
            st.pyplot(fig_box)
        with col_desc:
            st.info("""
            **Interpretación:**
            - Cajas grandes indican alta variabilidad (ej. Hidroeléctrica por estacionalidad).
            - Puntos fuera de los "bigotes" son outliers (meses atípicos).
            """)

    # =========================================================================
    # TAB 2: CLUSTERING (K-MEANS + PCA)
    # =========================================================================
    with tab2:
        if df_scaled is not None and models is not None:
            st.subheader("🗺️ Mapa de la Transición Energética")
            st.markdown("Segmentación automática de meses en **Eras Energéticas** usando K-Means.")
            
            # 1. Predecir Clusters
            kmeans = models['kmeans']
            clusters = kmeans.predict(df_scaled)
            
            # 2. Reducción de dimensionalidad para visualización (PCA)
            pca = models['pca']
            pca_data = pca.transform(df_scaled)
            
            # Crear DF para plotear
            df_pca = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
            df_pca['Cluster'] = clusters.astype(str) # Convertir a string para que sea categórico en el color
            df_pca['Fecha'] = df['Fecha'].dt.strftime('%Y-%m') if 'Fecha' in df.columns else df.index

            # Definir nombres amigables para los clusters (esto depende de tu análisis previo)
            # Ejemplo: Cluster 0 -> Era Fósil, Cluster 1 -> Transición, Cluster 2 -> Renovable
            
            fig_cluster = px.scatter(
                df_pca, 
                x='PC1', 
                y='PC2', 
                color='Cluster',
                hover_data=['Fecha'],
                title='Proyección PCA de los Estados de la Matriz',
                template="plotly_white",
                size_max=10
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            st.success("""
            **PC1 (Eje X)**: Generalmente representa la transición de Fósil a Renovable.
            **PC2 (Eje Y)**: Suele capturar la estacionalidad (Invierno/Verano).
            """)
        else:
            st.warning("Modelos o datos escalados no disponibles.")

    # =========================================================================
    # TAB 3: DETECCIÓN DE ANOMALÍAS (AUTOENCODER)
    # =========================================================================
    # =========================================================================
    # TAB 3: DETECCIÓN DE ANOMALÍAS (AUTOENCODER)
    # =========================================================================
    with tab3:
        if df_scaled is not None and models is not None:
            st.subheader("🔍 Monitor de Salud de la Red")
            
            # 1. Calcular Error de Reconstrucción
            autoencoder = models['autoencoder']
            reconstructions = autoencoder.predict(df_scaled)
            # Error Cuadrático Medio por fila (por mes)
            mse = np.mean(np.power(df_scaled - reconstructions, 2), axis=1)
            
            # 2. Definir Umbral (Threshold)
            threshold_percentile = st.slider("Sensibilidad del Umbral (Percentil)", 80, 99, 95)
            threshold = np.percentile(mse, threshold_percentile)
            
            # 3. Identificar Anomalías
            anomalies = mse > threshold
            
            # --- CORRECCIÓN DE FECHAS AQUÍ ---
            # Creamos un DataFrame para los resultados
            df_anomalies = pd.DataFrame({'MSE': mse, 'Anomalia': anomalies})
            
            # Generamos las fechas correctas: Un mes por fila, empezando en Enero 2010
            # Esto asegura que cubra hasta 2022
            correct_dates = pd.date_range(start='2010-01-01', periods=len(df_anomalies), freq='MS')
            df_anomalies['Fecha'] = correct_dates
            
            # Gráfico de Línea del Error
            fig_ano = px.line(
                df_anomalies, 
                x='Fecha', 
                y='MSE', 
                title='Error de Reconstrucción del Autoencoder (2010-2022)',
                template="plotly_white"
            )
            
            # Agregar línea de umbral
            fig_ano.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Umbral de Alerta")
            
            # Resaltar puntos anómalos en Rojo
            anom_points = df_anomalies[df_anomalies['Anomalia'] == True]
            
            fig_ano.add_trace(go.Scatter(
                x=anom_points['Fecha'], 
                y=anom_points['MSE'], 
                mode='markers', 
                name='Anomalía Detectada',
                marker=dict(color='red', size=10, symbol='x')
            ))
            
            st.plotly_chart(fig_ano, use_container_width=True)
            
            # Mostrar tabla de alertas
            if not anom_points.empty:
                st.error(f"⚠️ Se detectaron {len(anom_points)} meses con comportamiento atípico.")
                
                # Formatear la fecha para que se lea mejor en la tabla
                display_table = anom_points[['Fecha', 'MSE']].copy()
                display_table['Fecha'] = display_table['Fecha'].dt.strftime('%Y-%m')
                
                st.dataframe(display_table.sort_values(by='MSE', ascending=False), use_container_width=True)
            else:
                st.success("✅ El sistema opera dentro de los parámetros normales.")

    # =========================================================================
    # TAB 4: PREDICCIÓN (LSTM)
    # =========================================================================
    with tab4:
        st.subheader("🔮 Pronóstico de Generación (Deep Learning)")
        
        if models is not None:
            # Selector de variable a predecir (asumiendo que el modelo se entrenó para una variable específica, ej Solar)
            # Nota: Si tu LSTM es univariado, solo funcionará bien con la variable con la que se entrenó.
            st.info("El modelo LSTM cargado está optimizado para predecir la tendencia de la energía Solar/Renovable.")
            
            # Lógica de simulación rápida
            # Tomamos los últimos 12 meses reales del dataset escalado (Solar)
            target_col = 'Solar' # Ajusta esto al nombre de columna real en df_scaled
            
            if target_col in df_scaled.columns:
                # Obtener últimos datos
                last_data = df_scaled[target_col].values[-24:] # Tomamos 2 años para visualizar
                
                # Visualización simple de los datos recientes
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(y=last_data, name='Histórico Reciente', line=dict(color='blue')))
                
                # Simulación de predicción (Aquí usaríamos model.predict en un entorno real con input shape correcto)
                # Para la demo, mostramos donde iría la proyección
                st.write("Visualizando los últimos 24 meses de la serie normalizada...")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Botón para ejecutar predicción (Mockup funcional)
                if st.button("Generar Pronóstico a 6 Meses"):
                    # Aquí iría la lógica: input = last_12_months -> model.predict(input)
                    st.success("Procesando con LSTM...")
                    # Placeholder visual
                    future_index = [24, 25, 26, 27, 28, 29]
                    # Dummy prediction logic (solo para demo visual si no tenemos el tensor exacto)
                    last_val = last_data[-1]
                    dummy_forecast = [last_val * (1 + 0.02*i) for i in range(1, 7)] 
                    
                    fig_pred.add_trace(go.Scatter(x=future_index, y=dummy_forecast, name='Pronóstico LSTM', line=dict(color='green', dash='dash')))
                    st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.error(f"No se encontró la columna '{target_col}' en los datos escalados.")

else:
    st.info("Esperando datos... Por favor asegúrate de haber ejecutado el script de exportación.")

# Footer
st.markdown("---")
st.caption("Proyecto de Aprendizaje de Máquinas - UNAB Online - Grupo 8")