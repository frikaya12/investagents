"""World Bank Projects API : date d'approbation et variables structurées de chaque projet.

Endpoint vérifié par appels réels le 2026-11-15 : `https://search.worldbank.org/api/v2/projects`,
paramètres `format=json`, `rows` (1 000 testé), `os` (décalage), `fl` (champs), `id` (un projet).
Réponse : `{"total": ..., "projects": {"P123456": {...}, ...}}`. 22 729 projets au total.
Champs utilisés (formats observés) :
- `boardapprovaldate` : "2012-05-24T00:00:00Z" -> date d'approbation par le Conseil (référence
  anti-fuite) ;
- `approvalfy` : "2012" (texte) ; `closingdate` : "12/31/2024 12:00:00 AM" ;
- `totalamt` : "50,000,000" (texte avec virgules, USD) ; `lendinginstr`, `regionname`,
  `countryname` (liste).

Attention (pour la phase 2) : les montants de cette API sont les montants ACTUELS, qui peuvent
inclure des financements additionnels votés après l'approbation. Ce sont des variables à risque
de fuite ; à vérifier avant de les utiliser comme variables ex ante.

Lancer : `uv run python -m investagents.data.projects_api`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from investagents.data.http import get_json

API_URL = "https://search.worldbank.org/api/v2/projects"
PAGE_SIZE = 1000
FIELDS = (
    "id,boardapprovaldate,approvalfy,closingdate,totalamt,lendinginstr,"
    "regionname,countryname,status"
)

RAW_PATH = Path("data/raw/projects_raw.parquet")
PROCESSED_PATH = Path("data/processed/projects.parquet")


def fetch_all_projects(get=get_json) -> list[dict]:
    """Télécharge tous les projets, page par page."""
    rows: list[dict] = []
    offset = 0
    while True:
        page = get(API_URL, {"format": "json", "rows": PAGE_SIZE, "os": offset, "fl": FIELDS})
        projects = list(page.get("projects", {}).values())
        rows.extend(projects)
        offset += PAGE_SIZE
        if not projects or offset >= int(page["total"]):
            break
    return rows


def _to_amount(value) -> float | None:
    """'50,000,000' -> 50000000.0 ; vide ou illisible -> None."""
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def parse_projects(rows: list[dict]) -> pd.DataFrame:
    """Transforme les réponses brutes en table propre, une ligne par projet."""
    records = []
    for r in rows:
        countries = r.get("countryname") or []
        if isinstance(countries, str):
            countries = [countries]
        records.append(
            {
                "project_id": r.get("id"),
                "approval_date": r.get("boardapprovaldate"),
                "approval_fy_api": r.get("approvalfy"),
                "closing_date": r.get("closingdate"),
                "total_amount_usd": _to_amount(r.get("totalamt")),
                "lending_instrument": r.get("lendinginstr"),
                "region": r.get("regionname"),
                "country": "; ".join(countries) if countries else None,
                "status": r.get("status"),
            }
        )
    df = pd.DataFrame.from_records(records)
    # Date ISO avec « Z » -> date (jour) sans fuseau ; texte illisible -> manquant.
    df["approval_date"] = pd.to_datetime(df["approval_date"], utc=True, errors="coerce")
    df["approval_date"] = df["approval_date"].dt.tz_localize(None).dt.normalize()
    df["closing_date"] = pd.to_datetime(
        df["closing_date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    df["approval_fy_api"] = pd.to_numeric(df["approval_fy_api"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["project_id"]).drop_duplicates("project_id").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge la World Bank Projects API.")
    parser.add_argument("--refresh", action="store_true", help="ignorer le cache local")
    args = parser.parse_args()

    if RAW_PATH.exists() and not args.refresh:
        raw = pd.read_parquet(RAW_PATH)
        print(f"Cache utilisé : {RAW_PATH}")
    else:
        rows = fetch_all_projects()
        # Les champs liste (countryname) sont stockés en JSON texte pour un Parquet simple.
        raw = pd.DataFrame(rows).astype(str)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(RAW_PATH, index=False)
        print(f"Téléchargé : {len(raw)} projets -> {RAW_PATH}")
        df = parse_projects(rows)
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH, index=False)
    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Projets : {len(df)}  |  avec date d'approbation : {df['approval_date'].notna().sum()}")
    print(f"Écrit : {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
