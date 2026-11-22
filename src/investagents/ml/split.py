"""Coffre-fort et split temporel.

Coffre-fort : projets évalués par l'IEG en exercice ≥ 2025. Ils sont retirés AVANT toute autre
opération et ne servent ni à l'entraînement, ni au réglage, ni à l'évaluation, jusqu'à décision
d'Aya. `load_modeling_table` ne renvoie jamais une ligne du coffre.

Split temporel sur l'exercice d'approbation (justification dans reports/explication_phase2.md) :
- entraînement : approbation ≤ 2007 ;
- validation (réglage des hyperparamètres) : 2008–2011 ;
- test : 2012–2017 ;
- approbation ≥ 2018 : écartée (cohorte incomplète et biaisée : surtout des DPF évalués vite).
Après le réglage, le modèle final est réentraîné sur entraînement + validation (≤ 2011).
"""

from __future__ import annotations

import pandas as pd

VAULT_MIN_EVALUATION_FY = 2025
TRAIN_MAX_FY = 2007
VALID_FY = (2008, 2011)
TEST_FY = (2012, 2017)


def vault_mask(ieg: pd.DataFrame) -> pd.Series:
    return ieg["evaluation_fy"] >= VAULT_MIN_EVALUATION_FY


def assign_split(approval_fy: pd.Series) -> pd.Series:
    fy = approval_fy.astype(int)
    split = pd.Series("excluded", index=approval_fy.index)
    split[fy <= TRAIN_MAX_FY] = "train"
    split[(fy >= VALID_FY[0]) & (fy <= VALID_FY[1])] = "valid"
    split[(fy >= TEST_FY[0]) & (fy <= TEST_FY[1])] = "test"
    return split


def load_modeling_table(
    ieg_path: str = "data/processed/ieg_ratings.parquet",
    amounts_path: str = "data/processed/amounts_at_approval.parquet",
    with_country_indicators: bool = False,
    vault: bool = False,
) -> pd.DataFrame:
    """Projets notés, hors coffre-fort, joints au montant à l'approbation, avec `split`.

    `vault=True` : renvoie AU CONTRAIRE uniquement les projets du coffre-fort (évaluation finale
    unique, demandée par Aya ; experiments/vault/). Par défaut, jamais une ligne du coffre.

    `with_country_indicators` : ajoute les indicateurs WDI / CPIA / WGI (valeur d'une année
    ≤ approbation − 2), voir data/country_indicators.py.
    """
    ieg = pd.read_parquet(ieg_path)
    ieg = ieg[ieg["satisfactory"].notna()]
    ieg = ieg[vault_mask(ieg)] if vault else ieg[~vault_mask(ieg)]
    amounts = pd.read_parquet(amounts_path)[["project_id", "amount_at_approval_usd"]]
    df = ieg.merge(amounts, on="project_id", how="left")
    if with_country_indicators:
        from investagents.data import country_indicators as ci

        countries = pd.read_parquet(ci.PROJECT_COUNTRY_PATH)[["project_id", "iso3"]]
        long = pd.read_parquet(ci.LONG_PATH)
        proj = df[["project_id", "approval_fy"]].merge(countries, on="project_id", how="left")
        df = df.merge(ci.as_of_join(proj, long), on="project_id", how="left")
    df["split"] = assign_split(df["approval_fy"])
    return df.reset_index(drop=True)
