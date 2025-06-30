
# 🚀 Desafío Técnico ACQ 2025 - Data Scientist

Este proyecto aborda el desafío de construir un pipeline completo de análisis de datos y modelado predictivo, enfocado en productos de un marketplace. El objetivo es predecir si un producto es nuevo (`is_new`) y si será vendido (`sold_flag`), y extraer insights útiles para áreas de marketing y estrategia comercial.

---

## 🧭 Flujo de Trabajo del Proyecto

### 1. 🔍 Análisis Exploratorio de Datos (EDA)

El análisis exploratorio permite comprender la estructura y calidad de los datos antes del modelado. Se realizaron:

- Histogramas y boxplots para evaluar la distribución de variables numéricas.
- Mapas de calor para visualizar valores faltantes.
- Revisión de correlaciones entre variables.
  
Con base en los hallazgos, se decidió aplicar limpieza y transformación de columnas irrelevantes, así como ajustes a variables categóricas de baja calidad. Esta lógica fue implementada en la clase `RawCleaner`.

---

### 2. 🧱 Feature Engineering

Tras limpiar los datos, se generaron dos datasets específicos para clasificación binaria:

- `is_new`: ¿el producto es nuevo?
- `sold_flag`: ¿el producto tendrá al menos una venta?

La clase `FeatureEngineer`:
- Crea nuevas variables relevantes (frecuencias, flags, codificaciones ordinales).
- Calcula la **importancia de cada feature** con respecto a cada target utilizando Random Forest.

---

### 3. 🧠 Modelado Predictivo

Se entrenaron varios modelos clasificadores:

- XGBoost
- Gradient Boosting
- MLP Classifier
- Logistic Regression

La métrica principal utilizada fue **ROC AUC** porque:
- Evalúa la capacidad del modelo para distinguir entre clases.
- Es robusta frente a desbalance de clases.
- Ofrece una visión integral más allá del accuracy.

Posteriormente, se realizó **optimización de hiperparámetros** mediante GridSearchCV sobre los modelos mejor posicionados.

---

### 4. 📊 Análisis de Resultados

Resultados detallados de cada modelo, comparativas de métricas y visualización de importancia de variables están disponibles en:

📁 [`reports/analysis_results.md`](reports/analysis_results.md)

---

### 5. 💡 Insights para Marketing y Negocio

A partir de los modelos y las variables más predictivas, se generaron recomendaciones accionables para áreas como:

- Promociones
- Logística
- Gestión regional
- Segmentación de vendedores

Ver detalles en:

📁 [`reports/insights_marketing_negocio.md`](reports/insights_marketing_negocio.md)

---

### 6. ⚙️ Estrategia de Monitoreo e Implementación Técnica

Incluye recomendaciones sobre:

- Cómo versionar y monitorear el pipeline
- Registro de métricas de desempeño
- Estrategias de mantenimiento y reentrenamiento

Disponible en:

📁 [`reports/monitoring_pipelines.md`](reports/monitoring_pipelines.md)

---

## 🗂️ Estructura del Proyecto

- `src/`: Contiene las clases `DataAnalyzer`, `FeatureEngineer`, `ModelPredictor`.
- `data/`: Contiene el dataset original (`new_items_dataset.csv`).
- `reports/`: Reportes de métricas, análisis y resultados.
- `outputs/`: Outputs intermedios, como datasets transformados.

## Requisitos

- Python = 3.10
- pandas, numpy, scikit-learn, matplotlib, xgboost

Instalación rápida:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar
-Poner en data el csv data/new_items_dataset.csv

-Desde el directorio raíz:

```bash
python src/main.py
```

## Autor
[Carlos Sánchez]
