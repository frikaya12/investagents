"""Corpus des sections clés (8 000 caractères) de chaque dossier, avec et sans noms de lieux.

Pour chaque projet du corpus de la phase 2 (coffre-fort exclu par construction) :
- `sections` : sections clés extraites comme pour le comité (agents/sections.py), identifiants
  et nom du projet masqués ;
- `sections_masked_v1` : même texte, spaCy + pays des relevés IDA/IBRD remplacés par [PLACE] ;
- `sections_masked` (v2) : v1 + listes fixes GeoNames (pays, capitales, accents ignorés) et
  anciens noms (agents/masking.py).
Mis en cache dans data/processed/sections_corpus.parquet (non versionné).

Lancer : `uv run python -m investagents.data.sections_corpus` (tout reconstruire)
ou `... --remask` (réapplique seulement les listes à v1, sans relancer spaCy : même résultat,
car la v2 = listes appliquées après spaCy, et la liste v1 est incluse dans la v2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from investagents.agents.masking import list_mask, spacy_mask
from investagents.agents.sections import extract_key_sections, mask_identifiers, normalize
from investagents.data.corpus import CORPUS_INDEX_PATH, read_text
from investagents.ml.split import load_modeling_table

OUT_PATH = Path("data/processed/sections_corpus.parquet")
MAX_CHARS = 8000


def sections_text(raw: str, project_name: str | None) -> str:
    text = mask_identifiers(normalize(raw), name=project_name)
    return "\n".join(extract_key_sections(text, max_chars=MAX_CHARS).values())


def statement_country_names() -> set[str]:
    names = set()
    for s in ("ida", "ibrd"):
        names |= set(pd.read_parquet(f"data/raw/statement_{s}_raw.parquet")["country"].dropna())
    return names


def remask() -> None:
    out = pd.read_parquet(OUT_PATH)
    if "sections_masked_v1" not in out:  # premier passage : l'ancienne colonne devient v1
        out["sections_masked_v1"] = out["sections_masked"]
    out["sections_masked"] = list_mask(out.sections_masked_v1.tolist(), statement_country_names())
    out.to_parquet(OUT_PATH, index=False)
    print(f"Remasqué : {OUT_PATH} ({len(out)} projets)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remask", action="store_true")
    if parser.parse_args().remask:
        remask()
        return
    table = load_modeling_table()  # coffre-fort déjà exclu
    idx = pd.read_parquet(CORPUS_INDEX_PATH)
    docs = idx[idx.usable].merge(table[["project_id", "project_name"]], on="project_id")
    print(f"Documents : {len(docs)}", flush=True)
    texts = [sections_text(read_text(r.doc_id), r.project_name) for r in docs.itertuples()]
    print("Sections extraites ; masquage des lieux…", flush=True)
    names = statement_country_names()
    after_spacy = spacy_mask(texts)
    out = pd.DataFrame(
        {
            "project_id": docs.project_id,
            "sections": texts,
            "sections_masked_v1": list_mask(after_spacy, names, use_gazetteer=False),
            "sections_masked": list_mask(after_spacy, names),
        }
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Écrit : {OUT_PATH} ({len(out)} projets)")


if __name__ == "__main__":
    main()
