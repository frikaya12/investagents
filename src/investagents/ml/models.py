"""Modèles de la phase 2 : M0 (taux de base), M1 (gradient boosting), M1-texte (TF-IDF + LR).

Réglage des hyperparamètres : sur le jeu de VALIDATION (approbation 2008–2011), par AUC-PR.
Le jeu de test n'est jamais utilisé pour choisir quoi que ce soit. Le modèle final est ensuite
réentraîné sur entraînement + validation avec les hyperparamètres choisis.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

SEED = 42


# ---------------------------------------------------------------- M0
class BaseRate:
    """M0 : prédit pour tout projet le taux d'échec observé à l'entraînement."""

    def fit(self, x, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.rate_)
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------- M1
# Grille orientée vers une forte régularisation (demande d'Aya : réduire l'écart
# entraînement / validation). Arbres très courts, grosses feuilles, apprentissage lent,
# sous-échantillonnage des variables. Choix fait sur la validation seule.
M1_GRID = {
    "learning_rate": [0.01, 0.03, 0.1],
    "max_iter": [50, 100, 200],
    "max_depth": [1, 2, 3],
    "min_samples_leaf": [100, 300, 600],
    "l2_regularization": [1.0, 10.0],
    "max_features": [0.5, 1.0],
}


def make_m1(**params) -> HistGradientBoostingClassifier:
    # categorical_features="from_dtype" : les colonnes pandas « category » sont traitées comme
    # catégorielles nativement (pas d'encodage one-hot).
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", early_stopping=False, random_state=SEED, **params
    )


# ---------------------------------------------------------------- M1-texte
TEXT_MAX_CHARS = 150_000  # début du document : résumé, objectifs, financement, risques
M1_TEXT_GRID = {"C": [0.03, 0.1, 0.3, 1.0, 3.0]}


def make_m1_text(C: float = 1.0) -> Pipeline:
    """TF-IDF sur mots (lettres seulement : pas de chiffres, d'années ni d'identifiants)."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                    stop_words="english",
                    min_df=5,
                    max_df=0.9,
                    max_features=50_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            ("lr", LogisticRegression(C=C, max_iter=2000, solver="liblinear")),
        ]
    )


# ---------------------------------------------------------------- réglage
def tune(make, grid: dict, x_train, y_train, x_valid, y_valid) -> tuple[dict, pd.DataFrame]:
    """Essaie toutes les combinaisons ; renvoie la meilleure (AUC-PR en validation) + journal."""
    rows = []
    keys = list(grid)
    best, best_score = None, -np.inf
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        model = make(**params).fit(x_train, y_train)
        score = average_precision_score(y_valid, model.predict_proba(x_valid)[:, 1])
        rows.append({**params, "valid_auc_pr": score})
        if score > best_score:
            best, best_score = params, score
    log = pd.DataFrame(rows).sort_values("valid_auc_pr", ascending=False)
    return best, log
