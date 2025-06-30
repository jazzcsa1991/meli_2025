"""
feature_engineer.py

Este módulo define la clase `FeatureEngineer`, encargada de generar nuevas variables predictivas
a partir de datos crudos y calcular su importancia para tareas de clasificación.

Las variables generadas se orientan a modelos que predicen si un producto es nuevo (`is_new`)
y si tuvo ventas (`sold_flag`).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    """
    Clase encargada de realizar ingeniería de características sobre un dataset de productos.

    Esta clase transforma variables existentes, crea nuevas características relevantes para modelos predictivos,
    y calcula la importancia de las variables usando RandomForestClassifier.

    Atributos:
    ----------
    df : pd.DataFrame
        Dataset original sobre el que se realizarán transformaciones.
    feature_importance_summary : dict
        Diccionario que guarda la importancia de variables para distintos targets.
    """

    def __init__(self, df):
        """
        Inicializa el FeatureEngineer con una copia del DataFrame original.

        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame de entrada crudo.
        """
        self.df = df.copy()
        self.feature_importance_summary = {}

    def _generate_features(self, df):
        """
        Crea nuevas variables a partir de campos originales, agregando contexto útil para el modelado.

        Nuevas features generadas:
        - has_tags: indica si hay etiquetas disponibles
        - seller_product_count: cuántos productos vende el mismo seller
        - category_freq: frecuencia de la categoría en el dataset
        - seller_province_freq: frecuencia de la provincia del vendedor
        - seller_loyalty_encoded: codificación ordinal de la lealtad del vendedor
        - shipping_* columnas: booleanos transformados a enteros (0/1)

        Parámetros:
        -----------
        df : pd.DataFrame
            Dataset original sobre el que se crean las nuevas variables.

        Returns:
        --------
        pd.DataFrame
            Dataset con nuevas variables añadidas.
        """
        df = df.copy()
        
        # Indicador de si hay etiquetas disponibles
        df['has_tags'] = df['tags'].notnull().astype(int)
        
        # Conteo de productos por vendedor
        df['seller_product_count'] = df.groupby('seller_id')['seller_id'].transform('count')
        
        # Frecuencia de la categoría
        df['category_freq'] = df['category_id'].map(df['category_id'].value_counts())
        
        # Frecuencia de la provincia del vendedor
        df['seller_province_freq'] = df['seller_province'].map(df['seller_province'].value_counts())

        # Codificación ordinal de seller_loyalty
        loyalty_order = {'free': 0, 'bronze': 1, 'silver': 2, 'gold': 3, 'platinum': 4}
        df['seller_loyalty_encoded'] = df['seller_loyalty'].map(loyalty_order).fillna(-1)

        # Conversión de valores booleanos de texto a enteros
        for col in ['shipping_admits_pickup', 'shipping_is_free']:
            df[col] = df[col].map({'True': True, 'False': False})
            df[col] = df[col].fillna(False).astype(int)

        return df

    def _calculate_feature_importance(self, df, target_column, output_path):
        """
        Entrena un RandomForestClassifier para evaluar la importancia de las variables predictoras.

        Parámetros:
        -----------
        df : pd.DataFrame
            Dataset con features y variable objetivo.
        target_column : str
            Nombre de la variable objetivo a predecir.
        output_path : str
            Ruta donde guardar el CSV con la importancia de variables.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Separación train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Obtener importancia y guardar
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        self.feature_importance_summary[target_column] = importances
        importances.to_csv(output_path)

    def transform(self):
        """
        Ejecuta la ingeniería de características completa y crea datasets para dos tareas:

        - Clasificación de `is_new` (producto nuevo)
        - Clasificación de `sold_flag` (producto con ventas)

        Para cada uno, se calcula la importancia de las variables.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            - df_is_new: Dataset con target 'is_new'
            - df_sold: Dataset con target 'sold_flag'
        """
        # Generar nuevas features
        df_transformed = self._generate_features(self.df)

        # --- Dataset para clasificación de 'is_new'
        df_is_new = df_transformed.copy()
        df_is_new = df_is_new[[
            'price', 'has_tags', 'seller_product_count', 'category_freq',
            'seller_loyalty_encoded', 'shipping_admits_pickup', 'shipping_is_free', 'is_new'
        ]]
        self._calculate_feature_importance(df_is_new, 'is_new', 'outputs/is_new_feature_importance.csv')

        # --- Dataset para clasificación de 'sold_flag'
        df_sold = df_transformed.copy()
        df_sold['sold_flag'] = (df_sold['sold_quantity'] > 0).astype(int)
        df_sold = df_sold[[
            'price', 'has_tags', 'seller_product_count', 'category_freq',
            'seller_loyalty_encoded', 'shipping_admits_pickup', 'shipping_is_free', 'is_new', 'sold_flag'
        ]]
        self._calculate_feature_importance(df_sold, 'sold_flag', 'outputs/sold_flag_feature_importance.csv')

        return df_is_new, df_sold
