"""World Bank Documents & Reports API v3 : liste des documents d'un projet et accès au texte.

Comportement vérifié par appels réels le 2026-11-15 :
- endpoint `https://search.worldbank.org/api/v3/wds`, paramètres `format=json`, `rows`
  (jusqu'à 1 000 testé), `os` (décalage), `fl` (champs) ;
- `projectid=P122764` filtre par projet (113 documents pour ce projet) ;
- `docty_exact=Project Appraisal Document` filtre par type exact (7 005 documents) ; un type
  inexistant renvoie `total = 0` ;
- réponse : `{"total": "113", "documents": {"D123": {...}, ..., "facets": {...}}}` ;
- `projectid` d'un document peut lister plusieurs projets : "P514478,P518183" ;
- `docdt` : "2012-04-26T04:00:00Z" ; `lang` : "English", "Arabic", ... ;
- `txturl` : URL d'un fichier texte (texte du PDF), ex. 416 047 caractères pour le PAD de P122764.

Types ex ante repérés (comptes réels) : "Project Appraisal Document" (7 005),
"Staff Appraisal Report" (6 258, ancien nom), "Program Document" (1 608, prêts DPF),
"Project Information Document" (15 670, dont la version « Appraisal Stage »).

Lancer (échantillon de 50 projets notés) : `uv run python -m investagents.data.documents_api`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from investagents.data.http import get_json

API_URL = "https://search.worldbank.org/api/v3/wds"
PAGE_SIZE = 1000
FIELDS = "docty,docdt,projectid,txturl,pdfurl,lang,display_title,versiontyp,repnb"

SAMPLE_SIZE = 50
SEED = 42

RAW_SAMPLE_PATH = Path("data/raw/documents_sample50.json")
SAMPLE_PATH = Path("data/processed/documents_sample50.parquet")
SAMPLE_PROJECTS_PATH = Path("data/processed/sample50_projects.parquet")
TEXT_DIR = Path("data/raw/texts")


def fetch_documents(params: dict, get=get_json) -> list[dict]:
    """Toutes les pages d'une recherche ; chaque document garde son identifiant (`doc_id`)."""
    docs: list[dict] = []
    offset = 0
    while True:
        page = get(
            API_URL,
            {"format": "json", "rows": PAGE_SIZE, "os": offset, "fl": FIELDS, **params},
        )
        found = page.get("documents", {})
        batch = [{"doc_id": k, **v} for k, v in found.items() if k != "facets"]
        docs.extend(batch)
        offset += PAGE_SIZE
        if not batch or offset >= int(page["total"]):
            break
    return docs


def fetch_project_documents(project_id: str, get=get_json) -> list[dict]:
    """Tous les documents rattachés à un projet."""
    return fetch_documents({"projectid": project_id}, get=get)


def parse_documents(docs: list[dict]) -> pd.DataFrame:
    """Une ligne par (document, projet) : un document multi-projets est dupliqué par projet."""
    records = []
    for d in docs:
        pids = [p.strip() for p in str(d.get("projectid") or "").split(",") if p.strip()]
        for pid in pids or [None]:
            records.append(
                {
                    "doc_id": d.get("doc_id"),
                    "project_id": pid,
                    "docty": d.get("docty"),
                    "docdt": d.get("docdt"),
                    "lang": d.get("lang"),
                    "display_title": d.get("display_title"),
                    "versiontyp": d.get("versiontyp"),
                    "txturl": d.get("txturl") or None,
                    "pdfurl": d.get("pdfurl") or None,
                }
            )
    df = pd.DataFrame.from_records(
        records,
        columns=[
            "doc_id",
            "project_id",
            "docty",
            "docdt",
            "lang",
            "display_title",
            "versiontyp",
            "txturl",
            "pdfurl",
        ],
    )
    df["docdt"] = pd.to_datetime(df["docdt"], utc=True, errors="coerce")
    df["docdt"] = df["docdt"].dt.tz_localize(None).dt.normalize()
    return df


def sample_rated_projects(
    ieg: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = SEED
) -> pd.DataFrame:
    """Tirage aléatoire simple (graine fixe) parmi les projets notés (cible binaire connue)."""
    rated = ieg[ieg["satisfactory"].notna()]
    return rated.sample(n=n, random_state=seed).reset_index(drop=True)


def download_text(txturl: str, doc_id: str, *, text_dir: Path = TEXT_DIR, timeout_s: float = 120):
    """Télécharge le texte d'un document (mis en cache) et renvoie sa longueur en caractères.

    Renvoie 0 si l'URL ne répond pas (le document compte alors comme « sans texte »).
    """
    if not isinstance(txturl, str) or not txturl:  # None ou NaN (relu depuis Parquet)
        return 0
    path = text_dir / f"{doc_id}.txt"
    if path.exists():
        return len(path.read_text(encoding="utf-8"))
    try:
        resp = requests.get(txturl.replace("http://", "https://", 1), timeout=timeout_s)
    except requests.RequestException:
        return 0
    if resp.status_code != 200:
        return 0
    resp.encoding = "utf-8"
    text = resp.text.lstrip("﻿")
    text_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Liste les documents de 50 projets notés.")
    parser.add_argument("--refresh", action="store_true", help="ignorer le cache local")
    args = parser.parse_args()

    ieg = pd.read_parquet("data/processed/ieg_ratings.parquet")
    sample = sample_rated_projects(ieg)
    SAMPLE_PROJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PROJECTS_PATH, index=False)

    if RAW_SAMPLE_PATH.exists() and not args.refresh:
        raw = json.loads(RAW_SAMPLE_PATH.read_text(encoding="utf-8"))
        print(f"Cache utilisé : {RAW_SAMPLE_PATH}")
    else:
        raw = {}
        for i, pid in enumerate(sample["project_id"], 1):
            raw[pid] = fetch_project_documents(pid)
            print(f"[{i:2d}/{len(sample)}] {pid} : {len(raw[pid])} documents")
        RAW_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_SAMPLE_PATH.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    frames = []
    for pid, docs in raw.items():
        df = parse_documents(docs)
        # On ne garde que les lignes du projet interrogé (un document multi-projets est dupliqué).
        frames.append(df[df["project_id"] == pid])
    docs_df = pd.concat(frames, ignore_index=True)
    docs_df.to_parquet(SAMPLE_PATH, index=False)

    per_project = docs_df.groupby("project_id").size().reindex(sample["project_id"], fill_value=0)
    print(f"\n{len(docs_df)} lignes document-projet pour {len(sample)} projets")
    print(f"Projets sans aucun document : {(per_project == 0).sum()}")
    print("\nTypes de documents les plus fréquents :")
    print(docs_df["docty"].value_counts().head(20).to_string())
    print(f"\nÉcrit : {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
