# 💼 Insights para Marketing y Negocio

Este análisis se basa en la predicción de dos comportamientos clave del producto:

- 🎯 **`is_new`**: Indica si el producto listado es nuevo
- 📦 **`sold_flag`**: Indica si el producto tuvo al menos una venta

## 🎯 1. Estrategias de Promoción

- **Productos con envío gratuito (`shipping_is_free`) tienen mayor probabilidad de venta.**
  - Aparece como una de las variables más relevantes en la predicción de `sold_flag`.
  - Sugerencia: **Subvencionar el costo de envío** en campañas para productos con baja rotación.

- **El campo `has_tags` (información adicional visible) se relaciona positivamente con ventas.**
  - Sugerencia: Promover que los vendedores completen este campo para destacar sus publicaciones.

## 🌍 2. Enfoque Regional Inteligente

- **La variable `seller_province_freq` (frecuencia de publicaciones por provincia) influye en ventas.**
  - Algunas provincias muestran mayor efectividad comercial.
  - Sugerencia: **Priorizar campañas geolocalizadas** en regiones con alto `sold_flag`.

## 🧮 3. Segmentación de Vendedores

- **`seller_loyalty_encoded`** es una variable predictiva fuerte para `is_new`.
  - Vendedores con niveles altos de fidelidad (ej. *Gold*, *Platinum*) tienden a listar productos nuevos.
  - Sugerencia: Crear campañas exclusivas o anticipadas para estos segmentos.

- **`seller_product_count` también influye en las ventas.**
  - Vendedores con mayor volumen publicado tienen más probabilidad de conversión.
  - Sugerencia: Incentivar volumen en vendedores activos y brindar asistencia a los pequeños.

## 📦 4. Insights de Producto

- **El precio (`price`) se mantiene como variable clave en ambas predicciones (`is_new` y `sold_flag`).**
  - Precios extremos (outliers) se identificaron y excluyeron en la limpieza.
  - Sugerencia: Alinear políticas de precios según el segmento y comportamiento de venta.

- **La frecuencia de categoría (`category_freq`) es predictiva.**
  - Algunas categorías tienen más rotación que otras.
  - Sugerencia: Potenciar productos de alta conversión y reforzar los de baja performance.

## 🧪 5. Ideas de Casos de Uso Adicionales

| Caso de Uso                         | Técnica/Algoritmo Sugerido             |
|-------------------------------------|----------------------------------------|
| Clasificar si un producto se venderá| XGBoost / Gradient Boosting            |
| Segmentación de vendedores          | Clustering (K-Means, HDBSCAN)          |
| Optimización de visibilidad         | Modelos de uplift / análisis A/B       |
| Mejora de tags y contenido          | Análisis NLP / recomendaciones         |
| Predicción de nuevos productos      | Modelos de clasificación (`is_new`)    |

## 💡 Aplicación de los Insights

- 📢 **Campañas promocionales dirigidas** basadas en seller loyalty y región.
- 🚚 **Subsidios logísticos estratégicos** para productos con potencial y bajo envío.
- 📊 **Optimización del stock** y promociones según comportamiento histórico por categoría.
- 🎯 **Ajustes en reglas de visibilidad** en base a predicción de conversión (`sold_flag`).

