"""
model_predictor.py

Este módulo contiene la clase `ModelPredictor`, encargada de entrenar múltiples modelos
de clasificación sobre un dataset dado y evaluar su rendimiento.

Modelos incluidos por defecto:
- XGBoost
- Regresión Logística
- Gradient Boosting
- MLP (Perceptrón Multicapa)
- (Opcional) LightGBM

Las métricas evaluadas incluyen:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

Los resultados son devueltos en un DataFrame.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier  # Descomentar si se desea usar LightGBM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

class ModelPredictor:
    """
    Clase para entrenar y evaluar múltiples modelos clasificadores sobre un dataset dado.

    Atributos:
    ----------
    df : pd.DataFrame
        Dataset con variables predictoras y una columna objetivo.
    target_column : str
        Nombre de la columna que contiene la variable objetivo.
    models : dict
        Diccionario de modelos sklearn con su nombre como clave.
    results : list
        Lista para almacenar los resultados de evaluación por modelo.
    """

    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Inicializa el predictor con el dataset y la columna objetivo.

        Parámetros:
        -----------
        df : pd.DataFrame
            Dataset que contiene features y la columna objetivo.
        target_column : str
            Nombre de la columna a predecir.
        """
        self.df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # Eliminar columna fantasma si existe
        self.target_column = target_column
        self.models = {
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            # "LightGBM": LGBMClassifier(),  # Descomentar si LightGBM está instalado
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "GradientBoosting": GradientBoostingClassifier(),
            "MLP": MLPClassifier(max_iter=500)
        }
        self.results = []

    def train_and_evaluate(self):
        """
        Entrena y evalúa todos los modelos definidos en self.models.

        - Normaliza las features con StandardScaler.
        - Usa train_test_split con estratificación.
        - Calcula múltiples métricas de clasificación.

        Returns:
        --------
        pd.DataFrame:
            Tabla con métricas por modelo: Accuracy, Precision, Recall, F1, ROC AUC.
        """
        # Separar features y target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Normalización de features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # División del dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        # Entrenamiento y evaluación para cada modelo
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            report = classification_report(y_test, y_pred, output_dict=True)
            auc = roc_auc_score(y_test, y_prob)

            self.results.append({
                "Model": name,
                "Accuracy": report["accuracy"],
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1-Score": report["weighted avg"]["f1-score"],
                "ROC AUC": auc
            })

        return pd.DataFrame(self.results)
