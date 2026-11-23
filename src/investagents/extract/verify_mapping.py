"""Vérifie la table clé → projet du bloc D (experiments/blocD/sample.csv) pour TOUS les projets.

Pour chaque clé envoyée à Kaggle (data/processed/kaggle_blocD/projects.jsonl, limité aux clés
présentes dans les résultats du passage complet), on régénère en local les phrases candidates à
partir du document que la table associe à la clé, et on compare une empreinte SHA-256 de
(phrases numérotées + phrases candidates par facteur). Une seule différence signalerait une
étiquette fausse.

Lancer : `uv run python -m investagents.extract.verify_mapping`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from investagents.agents.sections import mask_identifiers, normalize
from investagents.committee_run.runner import load_jsonl
from investagents.data.corpus import read_text
from investagents.extract.extractor import candidates

SENT = Path("data/processed/kaggle_blocD/projects.jsonl")
SAMPLE = Path("experiments/blocD/sample.csv")
RESULTS = Path("experiments/blocD/kaggle_full1/results.jsonl")
OUT = Path("experiments/blocD/mapping_check.json")


def fingerprint(rec: dict) -> str:
    blob = json.dumps({"sentences": rec["sentences"], "by_factor": rec["by_factor"]},
                      sort_keys=True, ensure_ascii=False)  # fmt: skip
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main() -> None:
    keys = {r["key"] for r in load_jsonl(RESULTS) if r.get("factors")}
    sent = {}
    with SENT.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["key"] in keys:
                sent[r["key"]] = fingerprint(r)
            if len(sent) == len(keys):
                break
    sample = pd.read_csv(SAMPLE).set_index("key")
    exact, mismatches = 0, []
    for k in sorted(sent):
        row = sample.loc[k]
        text = mask_identifiers(normalize(read_text(row.doc_id)), name=row.project_name)
        if fingerprint(candidates(text)) == sent[k]:
            exact += 1
        else:
            mismatches.append(k)
    out = {"n_keys": len(keys), "n_found_in_sent_file": len(sent), "n_exact": exact,
           "mismatches": mismatches}  # fmt: skip
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
