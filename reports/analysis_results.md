# 📈 Análisis de Resultados

## 🔍 Modelos Evaluados

Se entrenaron y evaluaron los siguientes modelos de clasificación binaria para dos objetivos:

- Predicción de si un producto es **nuevo** (`is_new`)
- Predicción de si un producto fue **vendido** (`sold_flag`)

Modelos aplicados:

- **XGBoost Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**
- **MLP Classifier (red neuronal simple)**

Las métricas consideradas incluyen:

- Accuracy
- Precision (ponderada)
- Recall (ponderado)
- F1 Score (ponderado)
- ROC AUC

## 🧠 Resultados

### 📌 Predicción de `is_new`

| Modelo              | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| **XGBoost**          | 0.843    | 0.846     | 0.843  | 0.842    | **0.918** |
| Gradient Boosting   | 0.818    | 0.821     | 0.818  | 0.817    | 0.887   |
| MLP                 | 0.796    | 0.798     | 0.796  | 0.794    | 0.870   |
| Logistic Regression | 0.691    | 0.709     | 0.691  | 0.677    | 0.691   |

> **XGBoost fue seleccionado como el mejor modelo** para `is_new` por su mayor F1 Score y ROC AUC.

### 📌 Predicción de `sold_flag`

| Modelo              | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Gradient Boosting   | 0.856    | 0.838     | 0.856  | 0.836    | 0.837   |
| **XGBoost**          | 0.853    | 0.834     | 0.853  | 0.831    | **0.837** |
| MLP                 | 0.855    | 0.836     | 0.855  | 0.836    | 0.834   |
| Logistic Regression | 0.841    | 0.837     | 0.841  | 0.786    | 0.815   |

> En este caso, **Gradient Boosting superó ligeramente a XGBoost** en F1 Score y Accuracy.

---

## 🏆 Criterio de Selección

El modelo fue elegido principalmente con base en el **F1 Score ponderado**, ya que:
- Equilibra precisión y recall.
- Es robusto frente a clases desbalanceadas.
- Representa mejor la calidad general del modelo que Accuracy en estos casos.

---

## 🚀 Optimización de Modelos

Se aplicó **GridSearchCV** para afinar hiperparámetros de XGBoost y Gradient Boosting.

### `is_new` (mejorado con XGBoost)
- Accuracy: **0.85**
- F1 Score (ponderado): **0.85**
- ROC AUC: **0.9212**

### `sold_flag` (mejorado con Gradient Boosting)
- Accuracy: **0.85**
- F1 Score (ponderado): **0.83**
- ROC AUC: **0.8342**
- Observación: Recall bajo en clase positiva (33%) sugiere desbalance fuerte

---

## ⚠️ Limitaciones

- **Clase positiva ('sold_flag' = 1)** muy desbalanceada → afecta recall.
- **Valores faltantes** fueron imputados, pero pueden introducir sesgo.
- No se utilizaron campos de texto como `title`, que podrían mejorar el rendimiento.
- No se aplicó detección avanzada de colinealidad o selección de features automatizada.

---

## 📌 Próximos pasos

- Aplicar **estrategias de balanceo** (e.g. SMOTE, submuestreo).
- **Incluir procesamiento de texto** (`title`, `tags`) con NLP.
- Hacer **normalización más robusta** para modelos sensibles (e.g. redes neuronales).
- Probar modelos adicionales como **LightGBM**, **CatBoost**.
- Incorporar más **features contextuales** (competencia, demanda histórica, etc.).

