"""Variables structurées EX ANTE pour les modèles ML (M1), avec garde anti-fuite.

Une variable n'est admise que si elle est connue le jour de l'approbation du projet.

Interdites (règle de CLAUDE.md) : les colonnes IEG jugées après le projet
(`quality_at_entry`, `bank_performance`, `quality_of_supervision`, `me_quality`,
`final_closing_fy`, toute colonne `evaluation_*`), la cible et ses variantes, et par prudence :
- `country_fcs_status`, `country_fcs_lending_group`, `country_lending_group` : on ne sait pas à
  quelle date l'IEG mesure le statut du pays (il varie dans le temps pour 51 pays sur 187, sans
  que l'on sache s'il est pris à l'approbation ou à l'évaluation) ;
- `closing_date`, `total_amount_usd`, `status` de la Projects API : valeurs actuelles.

`assert_no_forbidden` est appelée par `build_features` : si une colonne interdite entre dans les
variables, le code s'arrête. Un test vérifie ce comportement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORBIDDEN_EXACT = frozenset(
    {
        # IEG, jugé après le projet
        "quality_at_entry",
        "bank_performance",
        "quality_of_supervision",
        "me_quality",
        "final_closing_fy",
        # cible et variantes
        "outcome",
        "outcome_rank",
        "satisfactory",
        "failure",
        # date de l'instantané IEG, nom du projet (identifiant)
        "as_of_date",
        "project_name",
        "project_id",
        # mesure du pays à une date incertaine (prudence)
        "country_fcs_status",
        "country_fcs_lending_group",
        "country_lending_group",
        # Projects API : valeurs actuelles
        "closing_date",
        "total_amount_usd",
        "status",
    }
)
FORBIDDEN_PREFIXES = ("evaluation_",)

# `country` retiré (décision d'Aya, 2026-11-15) : à performance égale sur le test (AUC-PR 0,268
# sans pays contre 0,270 avec), on garde le modèle le plus simple ; le pays (~180 modalités)
# augmentait surtout le surapprentissage. NB : sur la validation, le modèle avec pays faisait
# mieux (0,308 contre 0,280) ; cette décision a donc été prise après avoir vu le test.
CATEGORICAL = [
    "wb_region",
    "practice_group",
    "global_practice",
    "lending_instrument_type",
    "agreement_type",
]
NUMERIC = ["approval_fy", "log_amount_at_approval", "amount_missing"]
FEATURES = CATEGORICAL + NUMERIC


def is_forbidden(column: str) -> bool:
    return column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES)


def assert_no_forbidden(columns) -> None:
    bad = sorted(c for c in columns if is_forbidden(c))
    if bad:
        raise ValueError(f"Variables interdites (post-projet ou cible) : {bad}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice X (une ligne par projet, index = project_id) à partir de la table jointe.

    `df` doit contenir les colonnes IEG et `amount_at_approval_usd` (relevés IDA/IBRD).
    """
    x = pd.DataFrame(index=df["project_id"].to_numpy())
    x.index.name = "project_id"
    for col in CATEGORICAL:
        values = df[col].astype("string").str.strip().replace("", pd.NA).to_numpy()
        x[col] = pd.Categorical(values)
    x["approval_fy"] = df["approval_fy"].astype(float).to_numpy()
    amount = pd.to_numeric(df["amount_at_approval_usd"], errors="coerce").to_numpy(dtype=float)
    x["log_amount_at_approval"] = np.log1p(np.where(amount > 0, amount, np.nan))
    x["amount_missing"] = np.isnan(x["log_amount_at_approval"]).astype(float)
    assert_no_forbidden(x.columns)
    return x[FEATURES]


def target_failure(df: pd.DataFrame) -> pd.Series:
    """Cible des modèles : 1 = projet NON satisfaisant (MU/U/HU), classe minoritaire."""
    y = 1 - df["satisfactory"].astype(int)
    return pd.Series(y.to_numpy(), index=df["project_id"].to_numpy(), name="failure")
