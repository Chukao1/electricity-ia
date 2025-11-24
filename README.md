# ⚡ Sistema de Monitoreo y Predicción de la Transición Energética Chilena (2010-2022)

## 🌟 Resumen del Proyecto

Este proyecto implementa una arquitectura híbrida de Machine Learning y Deep Learning para analizar y pronosticar la evolución de la matriz energética del Sistema Eléctrico Nacional (SEN) de Chile, utilizando datos de generación mensual por fuente.

El análisis se centra en tres objetivos clave:
1.  **Predicción Supervisada (LSTM):** Pronóstico de generación de fuentes clave (ej. Solar, Hidro).
2.  **Caracterización Estructural (K-Means/PCA):** Identificación de "Eras Energéticas" a lo largo del tiempo.
3.  **Seguridad Operativa (Autoencoder):** Detección de meses con patrones de generación anómalos.

---

## 🛠️ Instalación y Dependencias

Este proyecto requiere Python 3.9+ y las librerías listadas a continuación. Se recomienda crear un entorno virtual (`conda` o `venv`) antes de la instalación.

### 1. Requisitos Principales

El siguiente comando instalará las dependencias críticas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
2. Librerías EspecíficasLas siguientes librerías son utilizadas para el modelado avanzado:pandas, numpy: Manejo y cálculo de datos.matplotlib, seaborn: Visualización de datos (EDA y resultados).scikit-learn: PCA, K-Means, Random Forest, y métricas (MAE, R2, MAPE).tensorflow / keras: Implementación de modelos Deep Learning (LSTM, Autoencoder).📂 Estructura del ProyectoEl proyecto sigue una estructura simple y modular:.
├── notebooks/
│   └── notebook.ipynb   # Notebook principal con todo el flujo de EDA y modelado.
├── data/
│   └── data.csv # Dataset original con datos de generación.
└── README.md                 # Este archivo.
▶️ Uso del ProyectoEl punto de entrada principal para reproducir el análisis es el notebook.Colocar los datos: Asegúrese de que el archivo data.csv se encuentre dentro de la carpeta data/.Iniciar el entorno: Active el entorno virtual donde instaló las dependencias.Ejecutar el Notebook: Abra notebooks/notebook.ipynb (o el nombre que haya dado a su notebook principal) y ejecute todas las celdas en orden cronológico.

Tareas Clave dentro del Notebook:
SecciónFunción PrincipalEDALimpieza de datos, pivoteo a matriz [periodo x producto], validación de VALUE.ClusteringDeterminación de k=3 óptimo (Codo/Silueta) e implementación de K-Means/PCA.
Deep LearningDefinición y entrenamiento de las arquitecturas LSTM y Autoencoder.
EvaluaciónCálculo de $R^2$, MAE, y MAPE para la selección del modelo final.


The dataset columns include:

COUNTRY: Name of the country
CODE_TIME: A code that represents the month and year (e.g., JAN2010 for January 2010)
TIME: The month and year in a more human-readable format (e.g., January 2010)
YEAR: The year of the data point
MONTH: The month of the data point as a number (1-12)
MONTH_NAME: The month of the data point as a string (e.g., January)
PRODUCT: The type of energy product (e.g., Hydro, Wind, Solar)
VALUE: The amount of electricity generated in gigawatt-hours (GWh)
DISPLAY_ORDER: The order in which the products should be displayed
yearToDate: The amount of electricity generated for the current year up to the current month in GWh
previousYearToDate: The amount of electricity generated for the previous year up to the current month in GWh
share: The share of the product in the total electricity generation for the country in decimal format