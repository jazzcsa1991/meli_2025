"""
raw_cleaner.py

Este módulo contiene la clase `RawCleaner`, cuyo propósito es aplicar una serie de reglas de
limpieza de datos sobre un DataFrame crudo obtenido de un marketplace o similar.

Las transformaciones incluyen:
- Eliminación de columnas irrelevantes
- Imputación de valores faltantes
- Conversión de tipos
- Normalización de campos booleanos
- Limpieza del campo 'tags'
- Eliminación de outliers en precios por categoría
"""

import pandas as pd
import numpy as np

class RawCleaner:
    """
    Clase para limpiar y transformar un DataFrame con datos crudos antes de análisis o modelado.

    Atributos:
    ----------
    df : pd.DataFrame
        Copia del DataFrame original, sobre el que se aplicarán las transformaciones.
    """

    def __init__(self, df):
        """
        Inicializa el limpiador con una copia del DataFrame original.

        Parámetros:
        -----------
        df : pd.DataFrame
            Datos sin procesar que serán limpiados.
        """
        self.df = df.copy()

    def clean(self):
        """
        Aplica múltiples reglas de limpieza y transformación de datos.

        Devuelve:
        ---------
        pd.DataFrame
            DataFrame limpio y transformado, listo para análisis o modelado.
        """

        # 1. Eliminar filas con demasiados valores faltantes (permite hasta 2 nulos)
        self.df.dropna(thresh=self.df.shape[1] - 2, inplace=True)

        # 2. Eliminar columnas irrelevantes o no útiles para el modelo
        to_drop = [
            'id', 'title', 'date_created', 'attributes', 'variations',
            'pictures', 'seller_country', 'seller_city', 'buying_mode',
            'shipping_mode', 'status', 'warranty', 'base_price',
            'available_quantity', 'initial_quantity'
        ]
        self.df.drop(columns=[col for col in to_drop if col in self.df.columns], inplace=True)

        # 3. Eliminar 'sub_status' si existe (inconsistente y poco útil)
        if 'sub_status' in self.df.columns:
            self.df.drop(columns=['sub_status'], inplace=True)

        # 4. Imputación de valores categóricos faltantes con el string "desconocido"
        cat_fill_const = ['seller_province', 'shipping_admits_pickup', 'shipping_is_free', 'warranty']
        for col in cat_fill_const:
            if col in self.df.columns:
                self.df[col].fillna('desconocido', inplace=True)

        # 5. Imputación numérica con la mediana (para evitar sesgo por outliers)
        num_cols = ['base_price', 'price', 'initial_quantity', 'sold_quantity', 'available_quantity']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')  # coerción segura
                self.df[col].fillna(self.df[col].median(), inplace=True)

        # 6. Corregir cantidades negativas (disponibilidad nunca debe ser negativa)
        if 'available_quantity' in self.df.columns:
            self.df['available_quantity'] = self.df['available_quantity'].apply(lambda x: max(x, 0))

        # 7. Convertir cadenas booleanas a valores booleanos reales
        bool_map = {'True': True, 'False': False}
        for col in ['shipping_admits_pickup', 'shipping_is_free']:
            if col in self.df.columns:
                self.df[col] = self.df[col].map(bool_map).fillna(False)

        # 8. Limpieza del campo 'tags', convirtiendo listas (o strings de listas) en string plano
        if 'tags' in self.df.columns:
            def clean_tags(val):
                if isinstance(val, list):
                    return ", ".join(val)
                elif isinstance(val, str):
                    try:
                        import ast
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, list):
                            return ", ".join(map(str, parsed))
                    except Exception:
                        pass
                    return val
                else:
                    return ""

            self.df['tags'] = self.df['tags'].apply(clean_tags)

        # 9. Eliminar outliers en 'price' usando el método IQR por categoría
        def remove_price_outliers_by_category(df):
            clean_df = pd.DataFrame()
            for cat in df['category_id'].unique():
                subset = df[df['category_id'] == cat]
                q1 = subset['price'].quantile(0.25)
                q3 = subset['price'].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                filtered = subset[(subset['price'] >= lower) & (subset['price'] <= upper)]
                clean_df = pd.concat([clean_df, filtered], axis=0)
            return clean_df.reset_index(drop=True)

        self.df = remove_price_outliers_by_category(self.df)

        return self.df
