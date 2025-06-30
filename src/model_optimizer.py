"""
model_optimizer.py

Este módulo contiene la clase `ModelOptimizer`, que entrena y optimiza modelos predictivos
para dos tareas de clasificación:

1. Predecir si un producto es nuevo (`is_new`) usando XGBoost.
2. Predecir si un producto fue vendido (`sold_flag`) usando Gradient Boosting.

El proceso incluye búsqueda en malla (GridSearchCV) y evaluación del rendimiento con métricas.
Resultados se guardan como archivos CSV en la carpeta `outputs/`.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

class ModelOptimizer:
    """
    Clase para entrenar y optimizar modelos clasificadores para dos objetivos distintos.

    Atributos:
    ----------
    df_is_new : pd.DataFrame
        Dataset para el modelo de 'is_new'.
    df_sold : pd.DataFrame
        Dataset para el modelo de 'sold_flag'.
    """

    def __init__(self, df_is_new: pd.DataFrame, df_sold: pd.DataFrame):
        """
        Inicializa con los datasets necesarios para entrenamiento.

        Parámetros:
        -----------
        df_is_new : pd.DataFrame
            Dataset con features e indicador 'is_new'.
        df_sold : pd.DataFrame
            Dataset con features e indicador 'sold_flag'.
        """
        self.df_is_new = df_is_new
        self.df_sold = df_sold

    def optimize(self):
        """
        Ejecuta GridSearchCV para encontrar los mejores hiperparámetros para:
        - XGBoost (clasificación de 'is_new')
        - GradientBoosting (clasificación de 'sold_flag')

        Guarda los resultados de evaluación en archivos CSV.
        """
        # Separar features y target
        X_new = self.df_is_new.drop(columns=['is_new'])
        y_new = self.df_is_new['is_new']

        X_sold = self.df_sold.drop(columns=['sold_flag'])
        y_sold = self.df_sold['sold_flag']

        # División entrenamiento/prueba
        Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_sold, y_sold, test_size=0.2, random_state=42)

        # Hiperparámetros usados en el entrenamiento para XGBoost
        xgb_params = {
            'n_estimators': [300],
            'max_depth': [7],
            'learning_rate': [0.2],
            'subsample': [1.0],
            'colsample_bytree': [0.6],
            'gamma': [0],
            'min_child_weight': [1],
            'scale_pos_weight': [1]
        }

        # Alternativa comentada: grid más amplio para XGBoost
        """
        xgb_params = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 1, 5],
            'min_child_weight': [1, 3, 5],
            'scale_pos_weight': [1, 2, 5]
        }
        """

        # Hiperparámetros usados en el entrenamiento para GradientBoosting
        gb_params = {
            'n_estimators': [200],
            'learning_rate': [0.1],
            'max_depth': [5],
            'min_samples_split': [2],
            'min_samples_leaf': [1, 3, 5],
            'subsample': [1.0],
            'max_features': ['log2']
        }

        # Alternativa comentada: grid más amplio para GradientBoosting
        """
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        """

        # GridSearch para XGBoost (target: is_new)
        xgb_grid = GridSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            param_grid=xgb_params,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        xgb_grid.fit(Xn_train, yn_train)
        print("🔧 Mejor configuración encontrada para XGBoost:", xgb_grid.best_params_)
        best_xgb = xgb_grid.best_estimator_

        # GridSearch para GradientBoosting (target: sold_flag)
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(),
            param_grid=gb_params,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        gb_grid.fit(Xs_train, ys_train)
        print("🔧 Mejor configuración encontrada para GradientBoosting:", gb_grid.best_params_)
        best_gb = gb_grid.best_estimator_

        # Guardar resultados de evaluación
        self._save_results(Xn_test, yn_test, best_xgb, "outputs/is_new_optimized_results.csv")
        self._save_results(Xs_test, ys_test, best_gb, "outputs/sold_flag_optimized_results.csv")

        print("✅ Optimización completada y resultados guardados.")

    def _save_results(self, X_test, y_test, model, filename):
        """
        Evalúa el modelo entrenado y guarda el reporte de clasificación y AUC en un archivo CSV.

        Parámetros:
        -----------
        X_test : pd.DataFrame
            Conjunto de prueba.
        y_test : pd.Series
            Etiquetas verdaderas.
        model : sklearn estimator
            Modelo entrenado.
        filename : str
            Ruta para guardar el archivo CSV de resultados.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Reporte de métricas
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)

        # Guardar como CSV
        df_report = pd.DataFrame(report).T
        df_report['roc_auc'] = auc
        df_report.to_csv(filename)

        # Mostrar en consola
        print(f"\n📊 Resultados de {filename}:")
        print(classification_report(y_test, y_pred))
        print(f"🔹 ROC AUC: {auc:.4f}\n")
