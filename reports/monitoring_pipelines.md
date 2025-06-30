# 📡 Monitoreo e Implementación Técnica del Pipeline ML (`is_new` y `sold_flag`)

Este documento describe la arquitectura completa para la orquestación, entrenamiento, despliegue y monitoreo del pipeline de machine learning basado en Google Cloud Platform (GCP), incluyendo integración con **Excel como fuente de datos** y **GitLab para CI/CD**.

---

## 🔁 1. Ingesta de Datos (Landing)

### 🟢 Origen: Excel → BigQuery

- Los datos de productos son cargados desde un archivo **Excel** (u otra fuente estructurada).
- Se utilizan conectores automáticos o scripts de carga hacia **BigQuery**, donde se almacena la tabla `data_source`.

> Este paso representa la entrada cruda del sistema, no transformada.

---

## 🧹 2. Capa Raw

### 🔹 BigQuery + Dataform + Cloud IAM

- `Dataform` ejecuta transformaciones SQL iniciales:
  - Conversión de tipos
  - Estandarización de campos
- Se crea una tabla estructurada: `raw_table`.

> Aquí se aplica la lógica inicial equivalente a lo que luego se codifica en la clase `RawCleaner`.

---

## 🧪 3. Capa Curada

### 🔹 BigQuery + Dataform + SA

- En esta etapa se aplican:
  - Reglas de negocio
  - Filtrado de columnas irrelevantes
  - Codificaciones necesarias para ML
- Se generan:
  - `model_ready_tables` (listas para el entrenamiento)
  - `Predictions` (resultados históricos y actuales del modelo)

> Esta etapa aplica el equivalente a las clases `RawCleaner` y `FeatureEngineer`.

---

## 🤖 4. ML Pipeline (Entrenamiento y Predicción)

### 🧭 Orquestado vía Cloud Composer | Desplegado en Cloud Run

#### a) `Query execution` – Extracción de Datos
- Un microservicio en `Cloud Run` extrae los datasets curados desde BigQuery.
- El acceso está restringido mediante `SA Comercial` (Service Account).

#### b) `Training Model` – Entrenamiento
- Script Python entrena modelos con los datasets (`is_new`, `sold_flag`).
- Se utilizan:
  - XGBoost
  - Gradient Boosting
  - MLP
  - Logistic Regression
- El modelo se versiona con Cloud Build y se almacena en **Artifact Registry**.

#### c) `Prediction`
- El modelo hace inferencias sobre los datos actuales.
- Se guarda la salida en la tabla `Predictions` de BigQuery.

#### d) `Send Results`
- Resultados enviados a sistemas consumidores (dashboards, APIs, alertas).
- Alternativamente, puede enviarse a un bucket, Pub/Sub o vía API REST.

---

## 🔄 CI/CD: GitLab Integration

- El código Python de entrenamiento y predicción es versionado en **GitLab**.
- Se activa CI/CD:
  - Al hacer *push*, GitLab ejecuta pruebas, linters y despliegues automáticos.
  - El modelo puede ser redeployado automáticamente en Cloud Run o registrado con nueva versión.

---

## 📊 5. Visualización y Reportes

### 🔹 Looker + BigQuery + Cloud IAM

- Dashboards en **Looker** acceden a la tabla `Predictions`.
- Se visualizan:
  - Predicciones agregadas
  - Métricas de rendimiento
  - Comparaciones históricas

> Se controlan accesos mediante roles definidos en `Cloud IAM`.

---

## 📈 6. Monitoreo y Versionamiento de Modelos

### 🔹 Cloud Monitoring + Artifact Registry + Cloud Build

- **Cloud Monitoring** registra:
  - Logs de ejecución de Cloud Run
  - Errores en consultas
  - Latencia y fallos de predicción

- **Artifact Registry** almacena:
  - Versiones entrenadas de modelos
  - Artefactos asociados (features, configs)

- **Cloud Build**:
  - Encargado de registrar automáticamente nuevas versiones de modelos.
  - Puede integrarse con GitLab CI para crear una nueva versión al hacer *merge* en producción.

---

## 🧩 Componentes Clave Resumidos

| Componente           | Rol                                                                 |
|----------------------|----------------------------------------------------------------------|
| Excel                | Fuente original de datos                                             |
| BigQuery             | Almacenamiento y consulta rápida de datos                           |
| Dataform             | Orquestación de SQL, curación y versionado de tablas                |
| Cloud Composer       | Orquestador general (Airflow)                                       |
| Cloud Run            | Entorno serverless para ejecutar scripts de predicción y entrenamiento |
| Python Scripts       | Implementación de modelos, extracción de datos, envío de resultados |
| GitLab CI/CD         | Control de versiones y despliegue automático de modelos             |
| Cloud Monitoring     | Métricas de salud y ejecución del pipeline                          |
| Artifact Registry    | Versionado y almacenamiento de modelos ML                           |
| Looker               | Visualización para stakeholders                                     |

---

## 🚀 Beneficios de la Arquitectura

- **Escalable**: Cloud Run y BigQuery soportan grandes volúmenes sin rediseño.
- **Segura**: IAM segmenta el acceso a datos, predicciones y entrenamiento.
- **Reproducible**: Todo está versionado y automatizado vía GitLab CI/CD.
- **Monitoreada**: Logs, métricas y errores están centralizados en Cloud Monitoring.

---

## ✅ Conclusión

Esta arquitectura asegura un **ciclo completo de machine learning en producción**, desde la carga cruda hasta la visualización final, con monitoreo y versionado activo de modelos.

> Es un diseño robusto, listo para escalar, que promueve la automatización y la mantenibilidad a largo plazo.
