"""Chargement des données de la démo, mis en cache (aucun calcul de modèle, aucun appel LLM).

Tous les chiffres affichés viennent soit de `report.metrics.load()` (les mêmes JSON que le
rapport), soit des fichiers de résultats déjà calculés dans experiments/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from investagents.committee_run.runner import load_jsonl
from investagents.report.metrics import ROOT, Metrics, load

EXP = ROOT / "experiments"
CRITERIA = ("relevance", "economic_efficiency", "feasibility", "risk", "impact_sustainability")


@st.cache_resource
def metrics() -> Metrics:
    return load()


@st.cache_data
def resume_prof() -> str:
    text = (ROOT / "docs" / "resume_prof.md").read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.startswith(("<!--", "# ")))


@st.cache_resource
def committee() -> tuple[dict, pd.DataFrame]:
    """Sorties du comité (bloc C) et métadonnées des projets notés, avec p_succès du ML."""
    from investagents.app.project_sheet import load_all

    return load_all()


@st.cache_resource
def factors_by_project() -> dict[str, dict]:
    """Facteurs vérifiés du bloc D (passage complet), par identifiant de projet."""
    sample = pd.read_csv(EXP / "blocD/sample.csv")
    key_to_pid = dict(zip(sample.key, sample.project_id, strict=True))
    out = {}
    for r in load_jsonl(EXP / "blocD/kaggle_full1/results.jsonl"):
        if r.get("factors"):
            out[key_to_pid[r["key"]]] = r["factors"]
    return out


@st.cache_data
def factor_citations() -> dict:
    path = EXP / "demo/factor_citations.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def random_draws() -> dict:
    return json.loads((EXP / "demo/random_draws.json").read_text(encoding="utf-8"))


@st.cache_data
def frontier(split: str = "valid") -> dict:
    name = "frontier_valid.json" if split == "valid" else "frontier_test.json"
    return json.loads((EXP / "blocC" / name).read_text(encoding="utf-8"))


@st.cache_data
def decision_pool() -> pd.DataFrame:
    """Les projets de VALIDATION notés par le comité (même univers que la frontière α)."""
    from investagents.selection.frontier import load_pool

    return load_pool("valid", EXP / "blocC/scores.parquet", EXP / "mvp/predictions.parquet")


@st.cache_data
def select_portfolio(alpha: float, weights: tuple[float, ...], budget: float) -> pd.DataFrame:
    """Portefeuille choisi par le sac à dos (même code que la frontière α)."""
    from investagents.selection.frontier import combined_score, select

    pool = decision_pool()
    w = dict(zip(CRITERIA, weights, strict=True))
    return select(pool, combined_score(pool, alpha, w), budget)


@st.cache_data
def calibration_curve(n_bins: int = 6) -> pd.DataFrame:
    """Courbe de calibration de M1-texte au test (probabilités recalibrées sur la validation).

    Tranches de même effectif (quantiles), pour que chaque point repose sur assez de projets.
    """
    d = pd.read_parquet(EXP / "blocB/test_predictions.parquet")
    p, y = d.p_fail_m1_text.to_numpy(), d.y_fail.to_numpy()
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(bins):
        m = bins == b
        rows.append({"predit": p[m].mean(), "observe": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


@st.cache_data
def test_predictions() -> pd.DataFrame:
    return pd.read_parquet(EXP / "blocB/test_predictions.parquet")


@st.cache_data
def identical_citation_share() -> float:
    """Part des paires Pour / Contre (même critère, même projet) qui citent les mêmes phrases."""
    recs, _ = committee()
    n = same = 0
    for r in recs.values():
        args = r["debate"]["arguments"]
        ids = {(a["criterion"], a["stance"]): a.get("sentence_ids") for a in args}
        for c in {k[0] for k in ids}:
            n += 1
            same += ids.get((c, "pro")) == ids.get((c, "contra"))
    return same / n
