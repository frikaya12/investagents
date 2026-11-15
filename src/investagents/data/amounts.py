"""Montant engagé À L'APPROBATION, par projet (variable ex ante).

Pourquoi ce module : les montants de la Projects API (`totalamt`, `totalcommamt`) sont les
montants actuels et peuvent inclure des financements votés après l'approbation. On utilise donc
les relevés officiels des prêts et crédits, qui donnent le montant initial de chaque prêt :

- IDA Statement of Credits, Grants and Guarantees, dernier instantané : FinancesOne DS00001 /
  RS00001 (11 509 lignes au 2026-11-15), champ `original_principal_amount_us_` ;
- IBRD Statement of Loans and Guarantees, dernier instantané : FinancesOne DS00047 / RS00047
  (9 545 lignes), champ `original_principal_amount`.
Licence CC BY 4.0. Une ligne = un prêt ou un crédit, avec `project_id` et `board_approval_date`
("11-May-1961"). Formats vérifiés par appels réels.

Règle : montant à l'approbation = somme des montants initiaux des prêts/crédits du projet
approuvés à la PREMIÈRE date d'approbation du projet. Un prêt approuvé plus tard (financement
supplémentaire rattaché au même projet) est exclu. Les projets GEF, fonds fiduciaires, etc. ne
sont pas dans ces relevés : montant manquant.

Lancer : `uv run python -m investagents.data.amounts`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from investagents.data.http import get_json

API_URL = "https://datacatalogapi.worldbank.org/dexapps/fone/api/apiservice"
PAGE_SIZE = 1000
SOURCES = {
    # nom : (datasetId, resourceId, colonne du montant initial)
    "IDA": ("DS00001", "RS00001", "original_principal_amount_us_"),
    "IBRD": ("DS00047", "RS00047", "original_principal_amount"),
}
RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/amounts_at_approval.parquet")


def fetch_statement(dataset_id: str, resource_id: str, get=get_json) -> pd.DataFrame:
    rows: list[dict] = []
    skip = 0
    while True:
        page = get(
            API_URL,
            {
                "datasetId": dataset_id,
                "resourceId": resource_id,
                "top": PAGE_SIZE,
                "skip": skip,
                "type": "json",
            },
        )
        rows.extend(page["data"])
        skip += PAGE_SIZE
        if len(page["data"]) < PAGE_SIZE:
            break
    if page.get("count") is not None and len(rows) != page["count"]:
        raise ValueError(f"{dataset_id} : {len(rows)} lignes reçues, {page['count']} annoncées")
    return pd.DataFrame(rows)


def standardize(raw: pd.DataFrame, source: str, amount_col: str) -> pd.DataFrame:
    """Colonnes communes : project_id, source, approval_date, original_amount_usd."""
    out = pd.DataFrame(
        {
            "project_id": raw["project_id"],
            "source": source,
            "approval_date": pd.to_datetime(
                raw["board_approval_date"], format="%d-%b-%Y", errors="coerce"
            ),
            "original_amount_usd": pd.to_numeric(raw[amount_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["project_id"])


def amount_at_approval(loans: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par projet : somme des prêts approuvés à la première date d'approbation."""
    loans = loans.dropna(subset=["approval_date"])
    first = loans.groupby("project_id")["approval_date"].transform("min")
    initial = loans[loans["approval_date"] == first]
    out = (
        initial.groupby("project_id")
        .agg(
            amount_at_approval_usd=("original_amount_usd", "sum"),
            first_loan_approval_date=("approval_date", "min"),
            n_initial_loans=("original_amount_usd", "size"),
        )
        .reset_index()
    )
    later = loans[loans["approval_date"] > first].groupby("project_id").size()
    out["n_later_loans"] = out["project_id"].map(later).fillna(0).astype(int)
    return out


def main() -> None:
    frames = []
    for source, (ds, rs, col) in SOURCES.items():
        path = RAW_DIR / f"statement_{source.lower()}_raw.parquet"
        if path.exists():
            raw = pd.read_parquet(path)
            print(f"Cache utilisé : {path}")
        else:
            raw = fetch_statement(ds, rs)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.astype(str).to_parquet(path, index=False)
            print(f"{source} : {len(raw)} prêts/crédits -> {path}")
        frames.append(standardize(raw, source, col))
    amounts = amount_at_approval(pd.concat(frames, ignore_index=True))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    amounts.to_parquet(PROCESSED_PATH, index=False)
    print(f"Projets avec montant à l'approbation : {len(amounts)}")
    n_later = int((amounts.n_later_loans > 0).sum())
    print(f"  dont avec des prêts approuvés plus tard (exclus) : {n_later}")
    print(f"Écrit : {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
