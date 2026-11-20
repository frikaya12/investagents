"""Pilote du comité : 2 projets du jeu de test (2012–2017), modèle local qwen2.5:3b.

Projets tirés au hasard (graine fixe) parmi les projets du test qui ont un dossier d'évaluation
exploitable ET une date d'approbation connue (décision d'Aya). Jamais de projet du coffre-fort.
Le document est celui choisi par leakage.py (corpus de la phase 2). Résultats en JSON dans
experiments/mvp/committee/ (identifiant réel conservé côté fichier, jamais montré aux agents).

Lancer : `uv run python -m investagents.agents.run_pilot [--n 2]`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from investagents.data.corpus import CORPUS_INDEX_PATH, read_text
from investagents.graph.committee import build_committee
from investagents.ml.split import load_modeling_table

OUT_DIR = Path("experiments/mvp/committee")
SEED = 7


def pick_projects(n: int) -> pd.DataFrame:
    table = load_modeling_table()
    projects = pd.read_parquet("data/processed/projects.parquet")[["project_id", "approval_date"]]
    idx = pd.read_parquet(CORPUS_INDEX_PATH)
    pool = (
        table[table.split == "test"]
        .merge(projects, on="project_id")
        .merge(idx[idx.usable], on="project_id")
    )
    pool = pool[pool.approval_date.notna()]
    return pool.sample(n=n, random_state=SEED).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = build_committee()
    for i, row in pick_projects(args.n).iterrows():
        t0 = time.time()
        state = graph.invoke(
            {
                "masked_id": f"PROJET-{i + 1:03d}",
                "raw_text": read_text(row.doc_id),
                "project_name": row.project_name,
                "model": args.model,
            }
        )
        cites = state.get("citations", [])
        result = {
            "masked_id": f"PROJET-{i + 1:03d}",
            "project_id": row.project_id,  # gardé pour l'analyse, jamais envoyé au LLM
            "approval_fy": int(row.approval_fy),
            "global_practice": row.global_practice,
            "outcome_ieg": row.outcome,  # vérité terrain, pour l'affichage seulement
            "model": args.model,
            "seconds_total": round(time.time() - t0, 1),
            "sections_chars": sum(len(v) for v in state["sections"].values()),
            "sections": state["sections"],
            "debate": state.get("debate"),
            "judge": state.get("judge"),
            "citations": state.get("citations", []),
            "n_sentences": len(state.get("sentences", [])),
            "citation_valid_rate": (sum(c["valid"] for c in cites) / len(cites)) if cites else 0.0,
            "llm_stats": state.get("llm_stats", {}),
            "errors": state.get("errors", []),
        }
        path = OUT_DIR / f"{row.project_id}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"{result['masked_id']} ({row.project_id}) : {result['seconds_total']} s, "
            f"citations valides {result['citation_valid_rate']:.0%}, "
            f"erreurs {result['errors']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
