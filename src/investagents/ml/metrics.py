"""Métriques d'évaluation et intervalles de confiance par bootstrap.

Convention : `y` = 1 si le projet est NON satisfaisant (classe minoritaire, ~25 %), `p` =
probabilité prédite d'échec. L'AUC-PR (average precision) est calculée sur cette classe : c'est
ce qui intéresse un bailleur (repérer les projets qui vont échouer). Valeur d'un modèle au hasard
= taux d'échec du jeu de test. AUC-ROC : 0,5 au hasard. Brier : erreur quadratique moyenne des
probabilités (plus bas = mieux).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

METRICS = {
    "auc_pr": average_precision_score,
    "auc_roc": roc_auc_score,
    "brier": brier_score_loss,
}


def compute_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {name: float(f(y, p)) for name, f in METRICS.items()}


def _bootstrap_indices(n: int, y: np.ndarray, n_boot: int, seed: int):
    """Tirages avec remise ; on saute ceux qui n'ont qu'une classe (AUC indéfinie)."""
    rng = np.random.default_rng(seed)
    kept = 0
    while kept < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            kept += 1
            yield idx


def bootstrap_ci(y, p, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Valeur observée + IC percentile à 95 % pour chaque métrique."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    point = compute_metrics(y, p)
    samples = {k: [] for k in METRICS}
    for idx in _bootstrap_indices(len(y), y, n_boot, seed):
        for name, f in METRICS.items():
            samples[name].append(f(y[idx], p[idx]))
    out = {}
    for name in METRICS:
        lo, hi = np.quantile(samples[name], [alpha / 2, 1 - alpha / 2])
        out[name] = {"value": point[name], "ci_low": float(lo), "ci_high": float(hi)}
    return out


def permutation_pvalue(y, p, metric: str = "auc_pr", n_perm: int = 2000, seed: int = 0) -> dict:
    """Test « le modèle fait-il mieux que le hasard ? ».

    On mélange les étiquettes `y` (ce qui casse tout lien entre scores et issue) et on recalcule
    la métrique. p-valeur = part des mélanges qui font au moins aussi bien que le vrai score
    (avec correction +1 pour ne jamais annoncer p = 0).
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    f = METRICS[metric]
    observed = f(y, p)
    rng = np.random.default_rng(seed)
    null = np.array([f(rng.permutation(y), p) for _ in range(n_perm)])
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= observed).sum()) / (n_perm + 1)),
    }


def bootstrap_diff(y, p_a, p_b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Écart apparié (modèle A − modèle B) sur les mêmes tirages, avec IC à 95 %.

    Apparié = les deux modèles sont évalués sur exactement les mêmes projets à chaque tirage,
    ce qui donne un intervalle plus juste que de comparer deux IC séparés.
    """
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    point = {k: f(y, p_a) - f(y, p_b) for k, f in METRICS.items()}
    samples = {k: [] for k in METRICS}
    for idx in _bootstrap_indices(len(y), y, n_boot, seed):
        for name, f in METRICS.items():
            samples[name].append(f(y[idx], p_a[idx]) - f(y[idx], p_b[idx]))
    out = {}
    for name in METRICS:
        lo, hi = np.quantile(samples[name], [alpha / 2, 1 - alpha / 2])
        out[name] = {"value": float(point[name]), "ci_low": float(lo), "ci_high": float(hi)}
    return out
