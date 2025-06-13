# Gas Company Customer Segmentation and Prediction Project

This project demonstrates a complete data analysis workflow, from initial data exploration and preparation to predicting payment defaults and segmenting customers for a simulated gas distribution company. The project is divided into three main Jupyter notebooks:

1.  **Data Exploration (`1. Exploration.ipynb`)**: Loads, inspects, cleans, and merges multiple data sources to create a consolidated analytical DataFrame.
2.  **First Prediction (`2. First Prediction.ipynb`)**: Uses the analytical DataFrame to build a Machine Learning model that predicts the probability of an invoice being paid late.
3.  **Segmentation (`3. Clustering.ipynb`)**: Applies clustering techniques (PCA and K-Means) on customer-level aggregated features to identify distinct customer segments.

## Notebook Contents

### 1. Data Exploration (`1. Exploration.ipynb`)

This notebook covers the initial stages of data analysis:

* **Introduction**: Presentation of the problem and the data.
* **Environment Setup and Data Loading**:
    * Importing libraries (`pandas`, `matplotlib`, `seaborn`).
    * Downloading the data files (`clientes.parquet`, `facturas.parquet`, `precios_gas.parquet`, `recaudo.parquet`).
    * Loading the data into Pandas DataFrames.
* **Initial Data Inspection**:
    * Individual analysis of each DataFrame (`.info()`, `.head()`, `.isnull().sum()`, `.describe()`, unique value counts).
* **Data Merging**:
    * Joining `df_facturas` with `df_clientes` (customer information).
    * Joining with `df_precios_gas` (to calculate cost).
    * Joining with `df_recaudo` (payment information).
    * Selecting relevant columns and handling duplicates to create `df_analisis`.
* **Feature Engineering**:
    * Converting date columns to `datetime` format.
    * Calculating `Precio por Consumo` (Price per Consumption).
    * Calculating time differences (e.g., `Dias_Emision_PagoOportuno`).
    * Creating the binary variable `Mora` (late payment indicator).
* **Detailed Exploration of the Consolidated DataFrame (`df_analisis`)**:
    * General summary and review of null values.
    * Visualization of the distribution of numerical (histograms, boxplots) and categorical (bar charts) variables.
    * Correlation analysis among numerical variables (heatmap).
    * Exploration of relationships (e.g., average consumption by stratum and city, late payment rate by stratum and city).
    * Basic temporal analysis of consumption.
* **Exploration Conclusions and Next Steps**.

### 2. First Prediction (`2. First Prediction.ipynb`)

This notebook focuses on building a model to predict late payments:

* **Introduction**: Defining the prediction objective (the `Mora` variable).
* **Environment Setup and Data Loading**:
    * Importing libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
    * Loading the `df_analisis.parquet` DataFrame (result of the previous notebook).
* **Quick Review and Final Data Preparation**:
    * Initial inspection of `df_analisis`.
    * Handling potential duplicate `mora`/`Mora` columns.
* **Feature and Target Variable Selection**:
    * Defining the features for the model (e.g., `Consumo (m3)`, `Estrato`, `Precio por Consumo`).
    * Warning about and excluding features that would cause data leakage (e.g., `Dias_PagoOportuno_PagoReal`).
* **Preprocessing for the Model**:
    * Encoding the categorical variable `Estrato` into an ordinal numerical format.
    * Handling missing values (NaN) in the selected features (using `dropna()`).
    * Defining `X` (feature matrix) and `y` (target vector).
* **Data Splitting**:
    * Separating the data into training and testing sets (`train_test_split`), using stratification on the target variable.
* **Model Training**:
    * Selecting and training a `RandomForestClassifier` model.
    * Using `class_weight='balanced'` to handle potential class imbalance.
* **Model Evaluation**:
    * Making predictions on the test set.
    * Calculating and explaining evaluation metrics:
        * Accuracy.
        * Classification Report (precision, recall, F1-score).
        * Confusion Matrix (with visualization).
* **Feature Importance**:
    * Extracting and visualizing the importance of each feature according to the Random Forest model.
* **Discussion of Results and Next Steps**:
    * Interpreting the model's performance.
    * Limitations and considerations.
    * Suggestions for future improvements.

### 3. Segmentation (`3. Clustering.ipynb`)

This notebook focuses on grouping customers into segments with similar characteristics:

* **Introduction**: Objective of segmentation and techniques to be used (PCA, K-Means).
* **Environment Setup and Data Loading**:
    * Importing libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
    * Loading the `df_analisis.parquet` DataFrame.
* **Quick Data Review**.
* **Feature Aggregation at the Customer Level**:
    * Grouping `df_analisis` by `Numero de contrato`.
    * Calculating average metrics (consumption, cost, days between events, late payment rate) for each customer, creating `df_grouped`.
    * Incorporating the `Estrato socioeconomico` variable into `df_grouped`, resulting in `df_segmentacion`.
* **Preprocessing for PCA and Clustering**:
    * Converting `Estrato socioeconomico` to a numerical format (`Estrato_Num`).
    * Selecting numerical features for PCA.
    * Handling NaN values (using `dropna()`).
    * Scaling features using `StandardScaler`.
* **Principal Component Analysis (PCA)**:
    * Reducing the dimensionality of the scaled data to 2 principal components.
    * Creating `df_pca` with the components.
    * Analyzing the explained variance.
* **K-Means Clustering**:
    * **Elbow Method**: Determining an optimal number of clusters (K) by visualizing the inertia (SSE) for different values of K.
    * **Applying K-Means**: Grouping the data (in PCA space) using the selected optimal K (e.g., K=4).
    * Adding segment labels to `df_pca`.
* **Segment Profiling**:
    * Joining the segment labels with the customer-level aggregated features DataFrame (`df_segmentacion_cleaned`).
    * Calculating the average characteristics for each segment.
    * Visualizing the segment profiles (e.g., bar charts of average characteristics).
    * Analyzing the distribution of `Estrato socioeconomico` within each segment.
    * Interpreting and potentially assigning descriptive names to the segments.
* **Conclusions and Next Steps**:
    * Summary of the identified segments.
    * Limitations and considerations.
    * Suggestions for validation, detailed profiling, and strategic actions.

## How to Use

1.  **Clone the Repository (if applicable)**:
    ```bash
    # git clone [REPOSITORY_URL]
    # cd [REPOSITORY_NAME]
    ```
2.  **Environment**:
    * The notebooks are designed to be run in Google Colab.
    * The main dependencies are `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`. These come pre-installed in Colab.
3.  **Data Download**:
    * Each notebook includes `!wget` commands to download the necessary `.parquet` files from the specified GitHub repository.
4.  **Execution**:
    * Open each notebook (`.ipynb`) in Google Colab.
    * Run the cells in sequential order.
    * The first notebook (`1_Exploracion.ipynb`) generates the `df_analisis.parquet` file, which is used by the subsequent notebooks. However, notebooks 2 and 3 also download this pre-generated file for modularity.

## Data

The data used are files in Parquet format:

* `clientes.parquet`: Customer information.
* `facturas.parquet`: Invoice details.
* `precios_gas.parquet`: Gas prices.
* `recaudo.parquet`: Payment information.
* `df_analisis.parquet`: Consolidated and preprocessed DataFrame, the result of `1. Exploracion.ipynb`.

These files are downloaded automatically at the beginning of each relevant notebook.