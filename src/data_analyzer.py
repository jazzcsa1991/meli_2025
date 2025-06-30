"""
data_analyzer.py

Este módulo define la clase `DataAnalyzer`, diseñada para realizar análisis exploratorio de datos (EDA)
sobre datasets relacionados con productos, precios y categorías.

Incluye:
- Estadísticas descriptivas
- Detección de valores faltantes
- Detección de outliers globales y por categoría
- Visualizaciones automáticas (histogramas, boxplots, correlaciones)
- Exportación de gráficos a archivos

Requiere que los outputs se puedan guardar en la carpeta `outputs/`.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

class DataAnalyzer:
    """
    Clase para realizar análisis exploratorio de datos (EDA) sobre un dataset cargado desde un archivo CSV.

    Atributos:
    ----------
    df : pd.DataFrame
        El dataset cargado.

    Métodos:
    --------
    describe_prices_and_quantities()
        Retorna estadísticas descriptivas de columnas relacionadas con precio y cantidad.

    missing_values_summary()
        Retorna un resumen del número de valores faltantes por columna.

    detect_price_outliers()
        Detecta outliers en la columna 'price' usando el criterio IQR.

    detect_price_outliers_by_category()
        Detecta outliers en 'price' pero segmentado por 'category_id'.

    run_eda()
        Ejecuta análisis exploratorio general e incluye visualizaciones guardadas en archivos.
    """

    def __init__(self, path):
        """
        Inicializa el DataAnalyzer cargando el dataset desde un archivo CSV.

        Parámetros:
        -----------
        path : str
            Ruta al archivo CSV.
        """
        self.df = pd.read_csv(path)

    def describe_prices_and_quantities(self):
        """
        Retorna estadísticas descriptivas básicas sobre columnas numéricas clave.
        """
        return self.df[['base_price', 'price', 'initial_quantity', 'sold_quantity', 'available_quantity']].describe()

    def missing_values_summary(self):
        """
        Devuelve un resumen del número de valores faltantes por columna.
        """
        return self.df.isnull().sum().sort_values(ascending=False)

    def detect_price_outliers(self):
        """
        Detecta outliers en la columna 'price' usando el método IQR global.
        """
        q1 = self.df['price'].quantile(0.25)
        q3 = self.df['price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = self.df[(self.df['price'] < lower_bound) | (self.df['price'] > upper_bound)]
        return outliers

    def detect_price_outliers_by_category(self):
        """
        Detecta valores atípicos en la columna 'price' segmentando por cada categoría ('category_id'),
        utilizando el método del rango intercuartílico (IQR).

        Returns:
            pd.DataFrame: Filas con precios atípicos por categoría.
        """
        outliers = pd.DataFrame()
        for cat in self.df['category_id'].unique():
            subset = self.df[self.df['category_id'] == cat]
            q1 = subset['price'].quantile(0.25)
            q3 = subset['price'].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = pd.concat([outliers, subset[(subset['price'] < lower) | (subset['price'] > upper)]])
        return outliers

    def run_eda(self):
        """
        Ejecuta un análisis exploratorio completo sobre el dataset,
        mostrando resúmenes, estadísticas, valores faltantes, distribuciones y correlaciones.
        Guarda visualizaciones en la carpeta 'outputs/'.
        """
        print("\n🔍 Dimensiones del dataset:", self.df.shape)

        print("\n📋 Primeras filas del dataset:")
        print(self.df.head())

        print("\n📋 Últimas filas del dataset:")
        print(self.df.tail())

        print("\n📊 Info general:")
        self.df.info()

        print("\n📈 Estadísticas numéricas:")
        print(self.df.describe())

        print("\n📊 Estadísticas categóricas:")
        print(self.df.describe(include='object'))

        print("\n🔢 Valores únicos por columna:")
        print(self.df.nunique())

        # Valores faltantes
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'missing_count': missing, 'missing_percent': missing_percent})
        print("\n🚨 Valores faltantes:")
        print(missing_df[missing_df['missing_count'] > 0])

        # Heatmap de valores faltantes
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title("Mapa de calor de valores faltantes")
        plt.tight_layout()
        plt.savefig("outputs/missing_heatmap.png")
        plt.close()

        # Separar columnas numéricas y categóricas
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(include='object').columns.tolist()

        # Visualización por cada columna numérica
        for col in tqdm(num_cols, desc="📊 Generando gráficos numéricos"):
            plt.figure(figsize=(12, 4))

            # Histograma (con muestra máxima de 5000)
            sample = self.df[col].dropna()
            sample = sample.sample(n=min(5000, len(sample)), random_state=42)
            plt.subplot(1, 2, 1)
            sns.histplot(sample, kde=False)
            plt.title(f"Distribución de {col}")
            plt.xlabel(col)

            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot de {col}")
            plt.xlabel(col)

            plt.tight_layout()
            plt.savefig(f"outputs/{col}_hist_box.png")
            plt.close()

        # Countplot para columnas categóricas
        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(y=col, data=self.df, order=self.df[col].value_counts().index[:10])
            plt.title(f"Frecuencia de {col}")
            plt.tight_layout()
            plt.savefig(f"outputs/{col}_countplot.png")
            plt.close()

        # Matriz de correlación entre variables numéricas
        corr = self.df[num_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title("Matriz de correlación")
        plt.tight_layout()
        plt.savefig("outputs/correlation_matrix.png")
        plt.close()
