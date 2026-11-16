"""Calibration des probabilités : erreur de calibration (ECE) et recalibration de Platt.

- ECE (expected calibration error) : on range les prédictions dans 10 intervalles égaux de
  probabilité ; dans chacun on compare la probabilité moyenne prédite au taux observé ; l'ECE est
  la moyenne pondérée de ces écarts. 0 = parfaitement calibré.
- Platt : régression logistique à une variable sur le logit de la probabilité, apprise sur la
  VALIDATION uniquement, puis appliquée telle quelle au test. Elle corrige notamment la dérive du
  taux d'échec (28 % à l'entraînement, 23 % en validation).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def ece(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Apprise sur (y_valid, p_valid), appliquée ensuite à n'importe quelles probabilités."""

    def fit(self, y, p):
        self.lr_ = LogisticRegression(C=1e6).fit(_logit(p), np.asarray(y))
        return self

    def transform(self, p):
        return self.lr_.predict_proba(_logit(p))[:, 1]
