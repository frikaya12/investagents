"""Sélection multicritère à partir des 5 notes du Juge (démo, bloc C).

L'utilisateur choisit un poids par critère ; le score d'un projet est la moyenne pondérée de ses
5 notes, ramenée entre 0 et 1 : score = Σ w_c · note_c / (10 · Σ w_c). Le portefeuille maximise la
somme des scores sous budget (même sac à dos OR-Tools que la page Portefeuille, même contrainte
de 25 % du budget par pays). Le score n'est PAS une probabilité : c'est un classement guidé par
les poids de l'utilisateur. Les poids appris sur les données, c'est M3 (régression logistique),
qui sera réglé sur la validation et évalué une seule fois.

Par défaut la démo n'utilise que les projets de la VALIDATION : choisir des poids en regardant le
résultat sur le test reviendrait à régler sur le test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from investagents.schemas.committee import CRITERIA

RESULTS = Path("experiments/blocC/kaggle_run1/results.jsonl")
SAMPLE = Path("experiments/blocC/sample.csv")


def scores_table(results_path: Path = RESULTS, sample_path: Path = SAMPLE) -> pd.DataFrame:
    """Une ligne par projet noté : 5 notes, taux de citations valides et infos du projet."""
    rows = []
    for line in Path(results_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("judge"):
            continue
        row = {"key": r["key"], "citation_valid_rate": r.get("citation_valid_rate")}
        row.update({s["criterion"]: s["score"] for s in r["judge"]["scores"]})
        rows.append(row)
    scores = pd.DataFrame(rows, columns=["key", "citation_valid_rate", *CRITERIA])
    scores = scores.drop_duplicates("key", keep="last")
    sample = pd.read_csv(sample_path)
    return sample.merge(scores, on="key", how="inner")


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Moyenne pondérée des 5 notes (0–10) ramenée entre 0 et 1."""
    w = pd.Series({c: float(weights.get(c, 0.0)) for c in CRITERIA})
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Les poids doivent être positifs et non tous nuls")
    return (df[list(CRITERIA)] * w).sum(axis=1) / (10 * w.sum())
