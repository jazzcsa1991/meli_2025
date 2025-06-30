"""
Script principal para ejecutar un pipeline de ciencia de datos sobre un dataset de productos.

Etapas del pipeline:
1. Análisis exploratorio de datos (EDA)
2. Limpieza de datos
3. Ingeniería de características
4. Entrenamiento y evaluación de modelos para:
    - Predicción de si un producto es nuevo (`is_new`)
    - Predicción de si un producto fue vendido (`sold_flag`)
5. Optimización de modelos

Requiere que los módulos personalizados estén disponibles:
- data_analyzer.py
- raw_cleaner.py
- feature_engineer.py
- model_predictor.py
- model_optimizer.py

Genera archivos de salida en la carpeta `outputs/`.
"""

from data_analyzer import DataAnalyzer
from feature_engineer import FeatureEngineer
from model_predictor import ModelPredictor
from raw_cleaner import RawCleaner
from model_optimizer import ModelOptimizer

def main():
    """
    Función principal que ejecuta todas las etapas del pipeline de ciencia de datos.
    Se carga el dataset, se analiza, limpia, transforma, y se entrena un modelo predictivo.
    Finalmente, se realiza una optimización de modelos.
    """

    # 1. Cargar y analizar datos
    print("📥 Cargando y analizando datos...")
    analyzer = DataAnalyzer("data/new_items_dataset.csv")
    
    # Estadísticas básicas
    print("📊 Estadísticas de precios y cantidades:\n", analyzer.describe_prices_and_quantities())
    print("🧹 Resumen de valores faltantes:\n", analyzer.missing_values_summary())
    print("🚨 Outliers de precios (global):\n", analyzer.detect_price_outliers())
    print("🚨 Outliers de precios por categoría:\n", analyzer.detect_price_outliers_by_category())

    # EDA completo (opcional, puede ser descomentado si se desea generar visualizaciones)
    analyzer.run_eda()

    # 2. Limpieza de datos crudos
    print("🧼 Limpiando datos...")
    df = analyzer.df  # Extraer el dataframe cargado
    cleaned_df = RawCleaner(df).clean()  # Aplicar reglas de limpieza definidas

    # 3. Ingeniería de características
    print("🔧 Generando nuevas variables...")
    df_is_new, df_sold = FeatureEngineer(cleaned_df).transform()

    # 4. Entrenamiento y evaluación para 'is_new'
    print("\n🧠 Entrenando modelos para predecir 'is_new'...")
    predictor_is_new = ModelPredictor(df_is_new, target_column='is_new')
    results_is_new = predictor_is_new.train_and_evaluate()
    print("📈 Resultados para 'is_new':\n", results_is_new)
    results_is_new.to_csv("outputs/is_new_model_results.csv", index=False)

    # 5. Entrenamiento y evaluación para 'sold_flag'
    print("\n🧠 Entrenando modelos para predecir 'sold_flag'...")
    predictor_sold = ModelPredictor(df_sold, target_column='sold_flag')
    results_sold = predictor_sold.train_and_evaluate()
    print("📈 Resultados para 'sold_flag':\n", results_sold)
    results_sold.to_csv("outputs/sold_model_results.csv", index=False)

    # 6. Optimización de modelos (ej. hiperparámetros, selección de features, etc.)
    print("\n🚀 Ejecutando optimización de modelos...")
    optimizer = ModelOptimizer(df_is_new, df_sold)
    optimizer.optimize()


if __name__ == "__main__":
    main()
