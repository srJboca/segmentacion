# Proyecto de Segmentación y Predicción para Empresa de Gas

Este proyecto demuestra un flujo de trabajo completo de análisis de datos, desde la exploración inicial y preparación de los datos hasta la predicción de mora en pagos y la segmentación de clientes para una empresa simulada de distribución de gas. El proyecto se divide en tres notebooks de Jupyter principales:

1.  **Exploración de Datos (`1. Exploracion.ipynb`)**: Carga, inspección, limpieza y combinación de múltiples fuentes de datos para crear un DataFrame analítico consolidado.
2.  **Primera Predicción (`2. Primera prediccion.ipynb`)**: Uso del DataFrame analítico para construir un modelo de Machine Learning que predice la probabilidad de que una factura entre en mora.
3.  **Segmentación (`3. Segmentacion.ipynb`)**: Aplicación de técnicas de clustering (PCA y K-Means) sobre características agregadas a nivel de cliente para identificar distintos segmentos de clientes.

## Contenido de los Notebooks

### 1. Exploración de Datos (`1. Exploracion.ipynb`)

Este notebook cubre las etapas iniciales del análisis de datos:

* **Introducción**: Presentación del problema y los datos.
* **Configuración del Entorno y Carga de Datos**:
    * Importación de librerías (`pandas`, `matplotlib`, `seaborn`).
    * Descarga de los archivos de datos (`clientes.parquet`, `facturas.parquet`, `precios_gas.parquet`, `recaudo.parquet`).
    * Carga de los datos en DataFrames de Pandas.
* **Inspección Inicial de los Datos**:
    * Análisis individual de cada DataFrame (`.info()`, `.head()`, `.isnull().sum()`, `.describe()`, conteo de valores únicos).
* **Combinación (Merge) de los Datos**:
    * Unión de `df_facturas` con `df_clientes` (información del cliente).
    * Unión con `df_precios_gas` (para calcular el costo).
    * Unión con `df_recaudo` (información de pagos).
    * Selección de columnas relevantes y manejo de duplicados para crear `df_analisis`.
* **Ingeniería de Características**:
    * Conversión de columnas de fecha a formato `datetime`.
    * Cálculo de `Precio por Consumo`.
    * Cálculo de diferencias temporales (e.g., `Dias_Emision_PagoOportuno`).
    * Creación de la variable binaria `Mora` (indicador de pago tardío).
* **Exploración Detallada del DataFrame Consolidado (`df_analisis`)**:
    * Resumen general y revisión de valores nulos.
    * Visualización de la distribución de variables numéricas (histogramas, boxplots) y categóricas (diagramas de barras).
    * Análisis de correlación entre variables numéricas (heatmap).
    * Exploración de relaciones (e.g., consumo promedio por estrato y ciudad, tasa de mora por estrato y ciudad).
    * Análisis temporal básico del consumo.
* **Conclusiones de la Exploración y Próximos Pasos**.

### 2. Primera Predicción (`2. Primera prediccion.ipynb`)

Este notebook se enfoca en construir un modelo para predecir la mora en los pagos:

* **Introducción**: Definición del objetivo de predicción (variable `Mora`).
* **Configuración del Entorno y Carga de Datos**:
    * Importación de librerías (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
    * Carga del DataFrame `df_analisis.parquet` (resultado del notebook anterior).
* **Revisión Rápida y Preparación Final de Datos**:
    * Inspección inicial de `df_analisis`.
    * Manejo de posibles columnas `mora`/`Mora` duplicadas.
* **Selección de Características y Variable Objetivo**:
    * Definición de las características (features) para el modelo (e.g., `Consumo (m3)`, `Estrato`, `Precio por Consumo`).
    * Advertencia y exclusión de características que causarían fuga de datos (e.g., `Dias_PagoOportuno_PagoReal`).
* **Preprocesamiento para el Modelo**:
    * Codificación de la variable categórica `Estrato` a formato numérico ordinal.
    * Manejo de valores faltantes (NaN) en las características seleccionadas (usando `dropna()`).
    * Definición de `X` (matriz de características) e `y` (vector objetivo).
* **División de Datos**:
    * Separación de los datos en conjuntos de entrenamiento y prueba (`train_test_split`), usando estratificación por la variable objetivo.
* **Entrenamiento del Modelo**:
    * Selección y entrenamiento de un modelo `RandomForestClassifier`.
    * Uso de `class_weight='balanced'` para manejar posible desbalance de clases.
* **Evaluación del Modelo**:
    * Realización de predicciones sobre el conjunto de prueba.
    * Cálculo y explicación de métricas de evaluación:
        * Accuracy.
        * Reporte de Clasificación (precisión, recall, F1-score).
        * Matriz de Confusión (con visualización).
* **Importancia de las Características**:
    * Extracción y visualización de la importancia de cada característica según el modelo Random Forest.
* **Discusión de Resultados y Próximos Pasos**:
    * Interpretación del rendimiento del modelo.
    * Limitaciones y consideraciones.
    * Sugerencias para mejoras futuras.

### 3. Segmentación (`3. Segmentacion.ipynb`)

Este notebook se centra en agrupar clientes en segmentos con características similares:

* **Introducción**: Objetivo de la segmentación y técnicas a utilizar (PCA, K-Means).
* **Configuración del Entorno y Carga de Datos**:
    * Importación de librerías (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
    * Carga del DataFrame `df_analisis.parquet`.
* **Revisión Rápida de Datos**.
* **Agregación de Características a Nivel de Cliente**:
    * Agrupación de `df_analisis` por `Numero de contrato`.
    * Cálculo de métricas promedio (consumo, costo, días entre eventos, tasa de mora) para cada cliente, creando `df_grouped`.
    * Incorporación de la variable `Estrato socioeconomico` a `df_grouped`, resultando en `df_segmentacion`.
* **Preprocesamiento para PCA y Clustering**:
    * Conversión de `Estrato socioeconomico` a formato numérico (`Estrato_Num`).
    * Selección de características numéricas para PCA.
    * Manejo de valores NaN (usando `dropna()`).
    * Escalado de características usando `StandardScaler`.
* **Análisis de Componentes Principales (PCA)**:
    * Reducción de la dimensionalidad de los datos escalados a 2 componentes principales.
    * Creación de `df_pca` con los componentes.
    * Análisis de la varianza explicada.
* **K-Means Clustering**:
    * **Método del Codo**: Determinación de un número óptimo de clusters (K) visualizando la inercia (SSE) para diferentes valores de K.
    * **Aplicación de K-Means**: Agrupación de los datos (en el espacio PCA) usando el K óptimo seleccionado (e.g., K=4).
    * Adición de las etiquetas de segmento a `df_pca`.
* **Perfilado de Segmentos**:
    * Unión de las etiquetas de segmento con el DataFrame de características agregadas a nivel de cliente (`df_segmentacion_cleaned`).
    * Cálculo de las características promedio para cada segmento.
    * Visualización de los perfiles de segmento (e.g., diagramas de barras de características promedio).
    * Análisis de la distribución de `Estrato socioeconomico` dentro de cada segmento.
    * Interpretación y posible asignación de nombres descriptivos a los segmentos.
* **Conclusiones y Próximos Pasos**:
    * Resumen de los segmentos identificados.
    * Limitaciones y consideraciones.
    * Sugerencias para validación, perfilado detallado y acciones estratégicas.

## Cómo Usar

1.  **Clonar el Repositorio (si aplica)**:
    ```bash
    # git clone [URL_DEL_REPOSITORIO]
    # cd [NOMBRE_DEL_REPOSITORIO]
    ```
2.  **Entorno**:
    * Los notebooks están diseñados para ser ejecutados en Google Colab.
    * Las dependencias principales son `pandas`, `numpy`, `matplotlib`, `seaborn`, y `scikit-learn`. Estas vienen preinstaladas en Colab.
3.  **Descarga de Datos**:
    * Cada notebook incluye comandos `!wget` para descargar los archivos `.parquet` necesarios desde el repositorio de GitHub especificado en los notebooks.
4.  **Ejecución**:
    * Abrir cada notebook (`.ipynb`) en Google Colab.
    * Ejecutar las celdas en orden secuencial.
    * El primer notebook (`1_Exploración.ipynb`) genera el archivo `df_analisis.parquet` que es utilizado por los notebooks subsecuentes. Aunque los notebooks 2 y 3 también descargan este archivo pre-generado para modularidad.

## Datos

Los datos utilizados son archivos en formato Parquet:

* `clientes.parquet`: Información de los clientes.
* `facturas.parquet`: Detalles de las facturas.
* `precios_gas.parquet`: Precios del gas.
* `recaudo.parquet`: Información de pagos.
* `df_analisis.parquet`: DataFrame consolidado y preprocesado, resultado de `1. Exploracion.ipynb`.

Estos archivos son descargados automáticamente al inicio de cada notebook relevante.
