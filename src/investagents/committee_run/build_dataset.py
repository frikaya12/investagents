"""Jeu de données du bloc C (à envoyer sur Kaggle, en privé).

Échantillon : 300 projets de la validation (2008–2011) et 300 du test (2012–2017), tirés avec une
graine fixe et stratifiés par issue (même taux d'échec que la période). Seuls les projets avec un
dossier ex ante exploitable (choisi par leakage.py, corpus de la phase 2) ET une date
d'approbation connue. Coffre-fort exclu (load_modeling_table).

Ce qui part sur Kaggle (projects.jsonl) : une clé opaque (C0001…), la période (valid / test),
l'ordre de traitement et les phrases numérotées des sections clés, identifiants et nom du projet
masqués. RIEN d'autre : ni identifiant réel, ni nom, ni issue IEG. La table de correspondance
clé → projet reste en local (experiments/blocC/sample.csv).

Ordre de traitement alterné validation / test, pour que des résultats partiels soient utilisables
pour les deux périodes.

Lancer : `uv run python -m investagents.committee_run.build_dataset`
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import pandas as pd

from investagents.agents.sections import (
    extract_key_sections,
    mask_identifiers,
    normalize,
    number_sentences,
)
from investagents.data.corpus import CORPUS_INDEX_PATH, read_text
from investagents.ml.features import target_failure
from investagents.ml.split import load_modeling_table

SEED = 2026
N_PER_SPLIT = 300
OUT_DIR = Path("data/processed/kaggle_blocC")
SAMPLE_CSV = Path("experiments/blocC/sample.csv")
CODE_PACKAGES = ["agents", "schemas", "debate", "graph", "committee_run"]
PROJECT_ID = re.compile(r"\bP\d{6}\b")


def stratified_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n projets avec le même taux d'échec que le groupe (arrondi)."""
    n_fail = round(n * pool.y_fail.mean())
    fail = pool[pool.y_fail == 1].sample(n_fail, random_state=seed)
    ok = pool[pool.y_fail == 0].sample(n - n_fail, random_state=seed)
    return pd.concat([fail, ok]).sample(frac=1, random_state=seed)  # mélangé


def interleave(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            rows.append(a.iloc[i])
        if i < len(b):
            rows.append(b.iloc[i])
    return pd.DataFrame(rows).reset_index(drop=True)


def pick() -> pd.DataFrame:
    table = load_modeling_table()  # coffre-fort exclu
    table["y_fail"] = target_failure(table).to_numpy()  # série indexée par projet
    projects = pd.read_parquet("data/processed/projects.parquet")[["project_id", "approval_date"]]
    idx = pd.read_parquet(CORPUS_INDEX_PATH)
    pool = table.merge(projects, on="project_id").merge(idx[idx.usable], on="project_id")
    pool = pool[pool.approval_date.notna()]
    valid = stratified_sample(pool[pool.split == "valid"], N_PER_SPLIT, SEED)
    test = stratified_sample(pool[pool.split == "test"], N_PER_SPLIT, SEED)
    out = interleave(valid, test)
    out["order"] = range(len(out))
    out["key"] = [f"C{i + 1:04d}" for i in out.order]
    return out


def agent_sentences(raw: str, project_name: str | None) -> list[dict]:
    text = mask_identifiers(normalize(raw), name=project_name)
    return number_sentences(extract_key_sections(text))


def main() -> None:
    sample = pick()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    n_sent = []
    with (OUT_DIR / "projects.jsonl").open("w", encoding="utf-8") as f:
        for r in sample.itertuples():
            sentences = agent_sentences(read_text(r.doc_id), r.project_name)
            blob = " ".join(s["text"] for s in sentences)
            # garde anti-fuite : ni identifiant de projet ni nom du projet côté agents
            assert not PROJECT_ID.search(blob), r.key
            if isinstance(r.project_name, str) and len(r.project_name) > 5:
                assert r.project_name.lower() not in blob.lower(), r.key
            n_sent.append(len(sentences))
            rec = {"key": r.key, "split": r.split, "order": int(r.order), "sentences": sentences}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sample["n_sentences"] = n_sent
    cols = ["key", "order", "split", "project_id", "y_fail", "outcome", "approval_fy",
            "project_name", "doc_id", "n_sentences"]  # fmt: skip
    extra = [c for c in ("country", "wb_region", "global_practice", "amount_at_approval_usd")
             if c in sample]
    sample[cols + extra].to_csv(SAMPLE_CSV, index=False)

    # code du comité (paquet investagents réduit), extrait et importé sur Kaggle
    src = Path("src/investagents")
    with tarfile.open(OUT_DIR / "investagents_code.tar.gz", "w:gz") as tar:
        tar.add(src / "__init__.py", arcname="investagents/__init__.py")
        for pkg in CODE_PACKAGES:
            for path in (src / pkg).rglob("*"):
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                tar.add(path, arcname=f"investagents/{path.relative_to(src)}", recursive=False)
    summary = sample.groupby("split").agg(n=("key", "size"), taux_echec=("y_fail", "mean"))
    print(summary)
    print(f"Phrases par projet : médiane {pd.Series(n_sent).median():.0f}, min {min(n_sent)}")
    print(f"Écrit : {OUT_DIR} ; correspondance locale : {SAMPLE_CSV}")


if __name__ == "__main__":
    main()
