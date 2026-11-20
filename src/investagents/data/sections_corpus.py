"""Corpus des sections clés (8 000 caractères) de chaque dossier, avec et sans noms de lieux.

Pour chaque projet du corpus de la phase 2 (coffre-fort exclu par construction) :
- `sections` : sections clés extraites comme pour le comité (agents/sections.py), identifiants
  et nom du projet masqués ;
- `sections_masked` : même texte, pays / lieux / nationalités remplacés par [PLACE]
  (agents/masking.py).
Mis en cache dans data/processed/sections_corpus.parquet (non versionné).

Lancer : `uv run python -m investagents.data.sections_corpus`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from investagents.agents.masking import mask_places
from investagents.agents.sections import extract_key_sections, mask_identifiers, normalize
from investagents.data.corpus import CORPUS_INDEX_PATH, read_text
from investagents.ml.split import load_modeling_table

OUT_PATH = Path("data/processed/sections_corpus.parquet")
MAX_CHARS = 8000


def sections_text(raw: str, project_name: str | None) -> str:
    text = mask_identifiers(normalize(raw), name=project_name)
    return "\n".join(extract_key_sections(text, max_chars=MAX_CHARS).values())


def main() -> None:
    table = load_modeling_table()  # coffre-fort déjà exclu
    idx = pd.read_parquet(CORPUS_INDEX_PATH)
    docs = idx[idx.usable].merge(table[["project_id", "project_name"]], on="project_id")
    print(f"Documents : {len(docs)}", flush=True)
    texts = [sections_text(read_text(r.doc_id), r.project_name) for r in docs.itertuples()]
    print("Sections extraites ; masquage des lieux…", flush=True)
    names = set()
    for s in ("ida", "ibrd"):
        names |= set(pd.read_parquet(f"data/raw/statement_{s}_raw.parquet")["country"].dropna())
    masked = mask_places(texts, country_names=names)
    out = pd.DataFrame(
        {"project_id": docs.project_id, "sections": texts, "sections_masked": masked}
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Écrit : {OUT_PATH} ({len(out)} projets)")


if __name__ == "__main__":
    main()
