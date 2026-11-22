"""Jeu de données du bloc D (Kaggle, privé) : phrases candidates par facteur, dossier complet.

Projets : tous ceux qui ont un dossier ex ante exploitable (corpus de la phase 2, choisi par
leakage.py : daté strictement avant l'approbation) ET une date d'approbation connue ; coffre-fort
exclu. Ordre de traitement : validation (2008–2011), puis test (2012–2017), puis entraînement
(≤ 2007), chaque période mélangée avec une graine fixe. Le pilote = les 30 premiers (validation).

Ce qui part sur Kaggle (projects.jsonl) : clé opaque (D00001…), période, ordre, phrases candidates
(identifiants et nom du projet masqués). Ni identifiant réel, ni nom, ni issue IEG. La
correspondance clé → projet reste en local (experiments/blocD/sample.csv).

Lancer : `uv run python -m investagents.extract.build_dataset`
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import pandas as pd

from investagents.agents.sections import mask_identifiers, normalize
from investagents.data.corpus import CORPUS_INDEX_PATH, read_text
from investagents.extract.extractor import candidates
from investagents.extract.factors import GRID_PATH, grid_hash
from investagents.ml.features import target_failure
from investagents.ml.split import load_modeling_table

SEED = 2027
OUT_DIR = Path("data/processed/kaggle_blocD")
SAMPLE_CSV = Path("experiments/blocD/sample.csv")
CODE_PACKAGES = ["agents", "schemas", "debate", "committee_run", "extract"]
PROJECT_ID = re.compile(r"\bP\d{6}\b")
ORDER = ["valid", "test", "train"]


def pick() -> pd.DataFrame:
    table = load_modeling_table()  # coffre-fort exclu
    table["y_fail"] = target_failure(table).to_numpy()
    projects = pd.read_parquet("data/processed/projects.parquet")[["project_id", "approval_date"]]
    idx = pd.read_parquet(CORPUS_INDEX_PATH)
    pool = table.merge(projects, on="project_id").merge(idx[idx.usable], on="project_id")
    pool = pool[pool.approval_date.notna()]
    parts = [pool[pool.split == s].sample(frac=1, random_state=SEED) for s in ORDER]
    out = pd.concat(parts).reset_index(drop=True)
    out["order"] = range(len(out))
    out["key"] = [f"D{i + 1:05d}" for i in out.order]
    return out


def main() -> None:
    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    assert grid["sha256"] == grid_hash(), "la grille du code ne correspond pas à la grille figée"
    sample = pick()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    n_cand = []
    with (OUT_DIR / "projects.jsonl").open("w", encoding="utf-8") as f:
        for r in sample.itertuples():
            text = mask_identifiers(normalize(read_text(r.doc_id)), name=r.project_name)
            cand = candidates(text)
            blob = " ".join(s["text"] for s in cand["sentences"])
            assert not PROJECT_ID.search(blob), r.key  # garde anti-fuite
            if isinstance(r.project_name, str) and len(r.project_name) > 5:
                assert r.project_name.lower() not in blob.lower(), r.key
            n_cand.append(len(cand["sentences"]))
            rec = {"key": r.key, "split": r.split, "order": int(r.order), **cand}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sample["n_candidates"] = n_cand
    cols = ["key", "order", "split", "project_id", "y_fail", "approval_fy", "project_name",
            "doc_id", "n_candidates"]  # fmt: skip
    sample[cols].to_csv(SAMPLE_CSV, index=False)
    src = Path("src/investagents")
    with tarfile.open(OUT_DIR / "investagents_code.tar.gz", "w:gz") as tar:
        tar.add(src / "__init__.py", arcname="investagents/__init__.py")
        for pkg in CODE_PACKAGES:
            for path in (src / pkg).rglob("*"):
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                tar.add(path, arcname=f"investagents/{path.relative_to(src)}", recursive=False)
    print(sample.groupby("split").size())
    print(f"Phrases candidates par projet : médiane {pd.Series(n_cand).median():.0f}")
    print(f"Écrit : {OUT_DIR} ; correspondance locale : {SAMPLE_CSV}")


if __name__ == "__main__":
    main()
