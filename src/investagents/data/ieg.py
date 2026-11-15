"""Données IEG : notes de résultat des projets de la Banque mondiale (vérité terrain).

Source : « IEG World Bank Project Performance Ratings », © Banque mondiale, licence CC BY 4.0.
Accès : API FinancesOne (dataset DS00053, ressource RS00055), 1 000 lignes par page au plus.
Format vérifié par un appel réel le 2026-11-15 : 12 663 lignes, 21 colonnes, dont `project_id`
(P + 6 chiffres), `approval_fy` (entier), `evaluation_fy` (texte), `outcome` (texte, 6 niveaux
+ « Not Rated »).

Lancer : `uv run python -m investagents.data.ieg`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from investagents.data.http import get_json

API_URL = "https://datacatalogapi.worldbank.org/dexapps/fone/api/apiservice"
DATASET_ID = "DS00053"
RESOURCE_ID = "RS00055"
PAGE_SIZE = 1000  # maximum autorisé par l'API

RAW_PATH = Path("data/raw/ieg_ratings_raw.parquet")
PROCESSED_PATH = Path("data/processed/ieg_ratings.parquet")

# Échelle IEG à 6 niveaux -> rang (1 = pire, 6 = meilleur) ; gardée pour l'analyse fine.
OUTCOME_SCALE = {
    "Highly Unsatisfactory": 1,
    "Unsatisfactory": 2,
    "Moderately Unsatisfactory": 3,
    "Moderately Satisfactory": 4,
    "Satisfactory": 5,
    "Highly Satisfactory": 6,
}
NOT_RATED = "Not Rated"


def fetch_ieg(get=get_json) -> pd.DataFrame:
    """Télécharge toutes les pages de l'API et renvoie les lignes brutes, sans transformation."""
    rows: list[dict] = []
    skip = 0
    while True:
        page = get(
            API_URL,
            {
                "datasetId": DATASET_ID,
                "resourceId": RESOURCE_ID,
                "top": PAGE_SIZE,
                "skip": skip,
                "type": "json",
            },
        )
        data = page["data"]
        rows.extend(data)
        skip += PAGE_SIZE
        if len(data) < PAGE_SIZE:
            break
    expected = page.get("count")
    if expected is not None and len(rows) != expected:
        raise ValueError(
            f"Téléchargement incomplet : {len(rows)} lignes reçues, {expected} annoncées"
        )
    return pd.DataFrame(rows)


def clean_ieg(raw: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le jeu IEG : types, cible binaire, une ligne par projet.

    - `outcome` : espaces retirés ; toute valeur hors échelle connue lève une erreur (on veut
      savoir si l'IEG change ses libellés, pas les ignorer).
    - `outcome_rank` : 1 à 6 ; `satisfactory` : 1 si MS/S/HS, 0 si MU/U/HU, manquant si non noté.
    - Doublons de `project_id` : on garde l'évaluation la plus récente
      (règle IEG « dernière note »).
    """
    df = raw.copy()
    df["outcome"] = df["outcome"].str.strip()
    unknown = set(df["outcome"].dropna()) - set(OUTCOME_SCALE) - {NOT_RATED}
    if unknown:
        raise ValueError(f"Libellés de note inconnus : {sorted(unknown)}")

    df["evaluation_fy"] = pd.to_numeric(df["evaluation_fy"], errors="raise").astype("Int64")
    df["approval_fy"] = df["approval_fy"].astype("Int64")
    df["final_closing_fy"] = pd.to_numeric(df["final_closing_fy"]).astype("Int64")

    df["outcome_rank"] = df["outcome"].map(OUTCOME_SCALE).astype("Int64")
    df["satisfactory"] = (df["outcome_rank"] >= 4).astype("Int64")
    df.loc[df["outcome_rank"].isna(), "satisfactory"] = pd.NA

    df = (
        df.sort_values(["project_id", "evaluation_fy"])
        .drop_duplicates("project_id", keep="last")
        .reset_index(drop=True)
    )
    return df


def summarize(df: pd.DataFrame) -> dict[str, pd.Series | int]:
    """Comptes demandés en phase 1 : projets, notes, années d'évaluation."""
    order = [*OUTCOME_SCALE, NOT_RATED]
    return {
        "n_projects": int(df["project_id"].nunique()),
        "n_rated": int(df["satisfactory"].notna().sum()),
        "outcome_counts": df["outcome"].value_counts().reindex(order, fill_value=0),
        "satisfactory_counts": df["satisfactory"].value_counts(dropna=False).sort_index(),
        "evaluation_fy_counts": df["evaluation_fy"].value_counts().sort_index(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-télécharger même si le cache existe"
    )
    args = parser.parse_args()

    if RAW_PATH.exists() and not args.refresh:
        raw = pd.read_parquet(RAW_PATH)
        print(f"Cache utilisé : {RAW_PATH}")
    else:
        raw = fetch_ieg()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(RAW_PATH, index=False)
        print(f"Téléchargé : {len(raw)} lignes -> {RAW_PATH}")

    df = clean_ieg(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    s = summarize(df)
    print(f"\nProjets distincts : {s['n_projects']}")
    print(f"Projets notés (hors « Not Rated ») : {s['n_rated']}")
    print("\nNote de résultat :\n" + s["outcome_counts"].to_string())
    print("\nCible binaire (1 = satisfaisant) :\n" + s["satisfactory_counts"].to_string())
    print("\nProjets par année d'évaluation :\n" + s["evaluation_fy_counts"].to_string())
    print(f"\nÉcrit : {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
