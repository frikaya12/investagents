"""Prépare deux petits fichiers pour la démo (aucune nouvelle expérience, aucun modèle modifié).

1. experiments/demo/random_draws.json : les 1 000 portefeuilles tirés au hasard du backtest figé
   (test, 25 % du budget), REPRODUITS avec le même code et la même graine que
   selection/choose_objective.py (seed = 2). Avant d'écrire, on vérifie que la moyenne et les
   quantiles 2,5 % / 97,5 % sont exactement ceux du JSON figé
   experiments/blocA/objective_test.json :
   c'est la même distribution, pas un nouveau tirage. Sert à l'histogramme « hasard contre notre
   portefeuille ».
2. experiments/demo/factor_citations.json : pour les projets de VALIDATION notés par le comité, le
   texte des phrases citées par les extracteurs (bloc D). Les résultats Kaggle ne gardent que les
   numéros de phrases ; le texte vient du jeu local data/processed/kaggle_blocD (non versionné,
   dossiers masqués). On ne garde que 2 phrases citées par facteur, 400 caractères au plus,
   comme les citations du comité déjà versionnées.

Lancer : `uv run python -m investagents.app.precompute`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from investagents.committee_run.runner import load_jsonl
from investagents.selection.choose_objective import load_pool, random_draws

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "demo"
OBJ_TEST = ROOT / "experiments/blocA/objective_test.json"
BLOC_C_SCORES = ROOT / "experiments/blocC/scores.parquet"
BLOC_D_SAMPLE = ROOT / "experiments/blocD/sample.csv"
BLOC_D_RESULTS = ROOT / "experiments/blocD/kaggle_full1/results.jsonl"
BLOC_D_TEXTS = ROOT / "data/processed/kaggle_blocD/projects.jsonl"
BUDGET = 0.25
MAX_PER_FACTOR = 2
MAX_CHARS = 400


def reproduce_random_draws() -> dict:
    frozen = next(r for r in json.loads(OBJ_TEST.read_text(encoding="utf-8"))["results"]
                  if r["budget"] == BUDGET)  # fmt: skip
    draws = random_draws(load_pool("test"), BUDGET, 1000, seed=2).count_share_satisfactory
    ref = frozen["random"]["count_share_satisfactory"]
    got = {"mean": float(draws.mean()), "p2_5": float(np.percentile(draws, 2.5)),
           "p97_5": float(np.percentile(draws, 97.5))}  # fmt: skip
    for k, v in got.items():
        if not np.isclose(v, ref[k], rtol=0, atol=1e-12):
            raise ValueError(f"tirages non reproduits : {k} = {v} contre {ref[k]} (JSON figé)")
    return {"budget": BUDGET, "seed": 2, "source": "selection/choose_objective.py, stage evaluate",
            "check": "moyenne et quantiles identiques à experiments/blocA/objective_test.json",
            "chosen_count_share": frozen["chosen"]["count_share_satisfactory"],
            "count_share": [round(float(x), 6) for x in draws]}  # fmt: skip


def factor_citations() -> dict:
    valid = pd.read_parquet(BLOC_C_SCORES).query("split == 'valid'").project_id
    d = pd.read_csv(BLOC_D_SAMPLE)
    keys = dict(zip(d.key, d.project_id, strict=True))
    wanted = {k for k, pid in keys.items() if pid in set(valid)}
    texts = {r["key"]: {s["id"]: s["text"] for s in r["sentences"]}
             for r in load_jsonl(BLOC_D_TEXTS) if r["key"] in wanted}  # fmt: skip
    out = {}
    for r in load_jsonl(BLOC_D_RESULTS):
        if r["key"] not in wanted or not r.get("factors"):
            continue
        sent = texts[r["key"]]
        # 2 premières phrases citées par facteur trouvé, 400 caractères au plus (extraits courts)
        ids = {i for f in r["factors"].values() if f["status"] == "found"
               for i in f["sentence_ids"][:MAX_PER_FACTOR]}  # fmt: skip
        out[keys[r["key"]]] = {"key": r["key"],
                               "sentences": {str(i): sent[i][:MAX_CHARS]
                                             for i in sorted(ids) if i in sent}}  # fmt: skip
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rd = reproduce_random_draws()
    (OUT / "random_draws.json").write_text(json.dumps(rd), encoding="utf-8")
    fc = factor_citations()
    (OUT / "factor_citations.json").write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(f"tirages reproduits : {len(rd['count_share'])} ; projets avec citations : {len(fc)}")


if __name__ == "__main__":
    main()
