# ⚡ Sistema de Monitoreo y Predicción de la Transición Energética Chilena (2010-2022)

## 🌟 Resumen del Proyecto

Este proyecto implementa una arquitectura híbrida de **Machine Learning** y **Deep Learning** para analizar, monitorear y pronosticar la evolución de la matriz energética del Sistema Eléctrico Nacional (SEN) de Chile. Se utilizan datos de generación mensual por fuente para entender la transición desde combustibles fósiles hacia energías renovables.

El análisis se estructura en tres pilares clave:
1.  **Predicción Supervisada (LSTM):** Redes neuronales recurrentes para el pronóstico de generación futura (ej. Solar, Hidro).
2.  **Caracterización Estructural (K-Means/PCA):** Segmentación no supervisada para identificar "Eras Energéticas" y visualizar la transición.
3.  **Seguridad Operativa (Autoencoder):** Detección de anomalías mediante la reconstrucción de patrones de generación, identificando meses atípicos.

---

## 🛠️ Instalación y Configuración Rápida

Para asegurar la reproducibilidad, se recomienda el uso de un entorno virtual. Sigue estos pasos para levantar el proyecto:

### 1. Clonar y crear entorno "grupo8"
```bash
git clone <URL_DE_TU_REPO>
cd proyecto_energia

# Windows
python -m venv grupo8
.\grupo8\Scripts\activate

# Mac / Linux
python3 -m venv grupo8
source grupo8/bin/activate

2. Instalar dependencias
Instala todas las librerías necesarias (Streamlit, TensorFlow, Plotly, etc.) con un solo comando:
pip install -r requirements.txt

▶️ Ejecución del Proyecto
El proyecto tiene dos componentes: el Entrenamiento (Notebook) y la Visualización (App).

Paso 1: Generar Modelos (Notebook)
Si es la primera vez que ejecutas el proyecto (o si la carpeta models/ está vacía), debes correr el notebook para entrenar las redes neuronales y procesar los datos.

Asegúrate de que data.csv esté en la carpeta data/.

Abre y ejecuta todas las celdas de notebooks/notebook.ipynb.

Esto generará los archivos .pkl y .h5 en la carpeta models/ y el dataset procesado.

Paso 2: Lanza el Dashboard (Streamlit)
Una vez entrenados los modelos, levanta la interfaz interactiva:
streamlit run app.py


📂 Estructura del Proyecto:
├── app.py                  # Frontend de visualización (Streamlit)
├── requirements.txt        # Lista de dependencias del entorno
├── notebooks/
│   └── notebook.ipynb      # Entrenamiento, EDA y validación de modelos
├── models/                 # Modelos serializados (generados por el notebook)
│   ├── lstm_model.h5
│   ├── autoencoder.h5
│   ├── kmeans_model.pkl
│   ├── pca_model.pkl
│   └── scaler.pkl
├── data/
│   ├── data.csv            # Dataset original (IEA)
│   └── data_processed.csv  # Datos limpios para el dashboard
└── grupo8/                 # Entorno Virtual (No se sube al repositorio)

📊 Descripción del Dataset
Los datos provienen de estadísticas mensuales de electricidad. Las columnas principales son:

COUNTRY: País de origen (Chile).

TIME: Fecha en formato legible (ej. "January 2010").

YEAR / MONTH: Desglose temporal numérico.

PRODUCT: Tipo de fuente energética (Hidráulica, Eólica, Solar, Carbón, etc.).

VALUE: Generación eléctrica en Gigavatios-hora (GWh).

share: Porcentaje de participación de la fuente en la matriz total.

yearToDate / previousYearToDate: Acumulados anuales para análisis de tendencias.

📚 Tecnologías Utilizadas
Frontend: Streamlit, Plotly (Gráficos interactivos).

Procesamiento: Pandas, NumPy, Scikit-learn (PCA, K-Means, Preprocesamiento).

Deep Learning: TensorFlow/Keras (LSTM, Autoencoder).

Visualización Estática: Matplotlib, Seaborn.