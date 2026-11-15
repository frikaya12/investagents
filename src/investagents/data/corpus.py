"""Corpus de textes pour M1-texte : un dossier d'évaluation par projet, filtré par leakage.py.

Choix du document : parmi les dossiers tier 1 (PAD, SAR, Program Document) gardés par
`filter_ex_ante` (anglais, daté strictement avant l'approbation, avec `txturl`), on prend le plus
récent : c'est la version finale soumise au Conseil.

Projets concernés : ceux de `load_modeling_table` (coffre-fort déjà exclu) dans les splits
train / valid / test. Aucun texte du coffre-fort n'est téléchargé.

Les textes sont stockés compressés (gzip) dans data/raw/texts_gz/ (non versionné).

Lancer : `uv run python -m investagents.data.corpus`
"""

from __future__ import annotations

import gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from investagents.data.leakage import filter_ex_ante
from investagents.ml.split import load_modeling_table

CENSUS_PATH = Path("data/processed/census_tier1_documents.parquet")
PROJECTS_PATH = Path("data/processed/projects.parquet")
TEXT_DIR = Path("data/raw/texts_gz")
CORPUS_INDEX_PATH = Path("data/processed/corpus_index.parquet")
MIN_CHARS = 10_000  # même seuil que la phase 1
N_THREADS = 4


def select_main_documents(kept: pd.DataFrame) -> pd.DataFrame:
    """Un document par projet : tier 1, avec txturl, le plus récent avant l'approbation."""
    d = kept[(kept["tier"] == 1) & kept["txturl"].notna()]
    d = d.sort_values(["project_id", "docdt", "doc_id"])
    return d.drop_duplicates("project_id", keep="last").reset_index(drop=True)


def text_path(doc_id: str, text_dir: Path = TEXT_DIR) -> Path:
    return text_dir / f"{doc_id}.txt.gz"


def read_text(doc_id: str, text_dir: Path = TEXT_DIR) -> str:
    with gzip.open(text_path(doc_id, text_dir), "rt", encoding="utf-8") as f:
        return f.read()


def download_gz(doc_id: str, txturl: str, text_dir: Path = TEXT_DIR, timeout_s: float = 120) -> int:
    """Télécharge un texte (si absent du cache) ; renvoie sa longueur, 0 en cas d'échec."""
    path = text_path(doc_id, text_dir)
    if path.exists():
        return len(read_text(doc_id, text_dir))
    for _ in range(3):
        try:
            resp = requests.get(txturl.replace("http://", "https://", 1), timeout=timeout_s)
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            resp.encoding = "utf-8"
            text = resp.text.lstrip("﻿")
            text_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(text)
            return len(text)
        if resp.status_code == 404:
            return 0
    return 0


def main() -> None:
    table = load_modeling_table()
    table = table[table["split"].isin(["train", "valid", "test"])]
    projects = pd.read_parquet(PROJECTS_PATH)[["project_id", "approval_date"]]
    approvals = table[["project_id", "approval_fy"]].merge(projects, on="project_id", how="left")
    census = pd.read_parquet(CENSUS_PATH)
    census = census[census["project_id"].isin(set(table["project_id"]))]
    docs = select_main_documents(filter_ex_ante(census, approvals))
    print(f"Projets à télécharger : {len(docs)}")

    def job(row):
        return download_gz(row.doc_id, row.txturl)

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        lengths = []
        for i, n in enumerate(pool.map(job, docs.itertuples()), 1):
            lengths.append(n)
            if i % 250 == 0:
                print(f"  {i}/{len(docs)}", flush=True)
    docs["n_chars"] = lengths
    docs["usable"] = docs["n_chars"] >= MIN_CHARS
    cols = ["project_id", "doc_id", "docty", "docdt", "approval_ref", "n_chars", "usable"]
    docs[cols].to_parquet(CORPUS_INDEX_PATH, index=False)
    print(f"Textes exploitables (≥ {MIN_CHARS} caractères) : {int(docs['usable'].sum())}")
    print(f"Écrit : {CORPUS_INDEX_PATH}")


if __name__ == "__main__":
    main()
