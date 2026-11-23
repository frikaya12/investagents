"""Fiche projet de la démo : probabilité de réussite (ML) + arguments des agents + notes.

Assemble, pour une clé du bloc C (C0001…), ce qui est déjà calculé (aucun appel LLM) :
- p_succès : M1-texte calibré (experiments/mvp/predictions.parquet) ;
- arguments Pour / Contre finaux par critère, avec les phrases citées (vérifiées) et le verdict
  du critique sur l'argument initial ;
- notes du Juge (0–10) et justifications ;
- facteurs extraits par le bloc D, s'ils existent pour ce projet (correspondance par projet,
  en local).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from investagents.committee_run.runner import load_jsonl

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/blocC/kaggle_run1/results.jsonl"
SAMPLE = ROOT / "experiments/blocC/sample.csv"
PRED = ROOT / "experiments/mvp/predictions.parquet"
BLOC_D_SAMPLE = ROOT / "experiments/blocD/sample.csv"
BLOC_D_RUNS = ROOT / "experiments/blocD"


def load_all(results=RESULTS, sample=SAMPLE, pred=PRED) -> tuple[dict, pd.DataFrame]:
    recs = {r["key"]: r for r in load_jsonl(Path(results)) if r.get("judge")}
    meta = pd.read_csv(sample)
    p = pd.read_parquet(pred)[["m1_text_cal"]]
    meta = meta.merge(p, left_on="project_id", right_index=True, how="left")
    meta["p_success"] = 1 - meta.m1_text_cal
    return recs, meta[meta.key.isin(recs)].reset_index(drop=True)


def load_factors(project_id: str) -> dict | None:
    """Facteurs du bloc D pour ce projet (dernier passage trouvé), sinon None."""
    if not BLOC_D_SAMPLE.exists():
        return None
    d = pd.read_csv(BLOC_D_SAMPLE)
    keys = d.loc[d.project_id == project_id, "key"]
    if keys.empty:
        return None
    for path in sorted(BLOC_D_RUNS.glob("kaggle_*/results.jsonl"), reverse=True):
        for r in load_jsonl(path):
            if r["key"] == keys.iloc[0] and r.get("factors"):
                return r["factors"]
    return None


def sheet(rec: dict, row: pd.Series, factors_lookup=load_factors) -> dict:
    """Contenu de la fiche d'un projet (`factors_lookup` : project_id -> facteurs, ou None)."""
    cites = {(c["criterion"], c["stance"]): c for c in rec.get("citations", [])}
    verdicts = {}
    initial = rec["initial_debate"]["arguments"]
    for h in rec.get("rounds_history", [])[:1]:
        for cr in h["critiques"]:
            a = initial[cr["argument_index"]]
            verdicts[(a["criterion"], a["stance"])] = cr["verdict"]
    args = []
    for a in rec["debate"]["arguments"]:
        k = (a["criterion"], a["stance"])
        c = cites.get(k, {})
        args.append({
            "criterion": a["criterion"],
            "stance": a["stance"],
            "claim": a["claim"],
            "cited": c.get("cited_text", []),
            "citation_valid": bool(c.get("valid")),
            "critic_verdict": verdicts.get(k),
        })  # fmt: skip
    return {
        "key": rec["key"],
        "split": row["split"],
        "country": row.get("country"),
        "sector": row.get("global_practice"),
        "amount_usd": row.get("amount_at_approval_usd"),
        "p_success": float(row["p_success"]),
        "scores": rec["judge"]["scores"],
        "arguments": args,
        "factors": factors_lookup(row["project_id"]),
        "outcome": row.get("outcome"),
    }


def main() -> None:  # aperçu en ligne de commande
    recs, meta = load_all()
    s = sheet(recs[meta.key.iloc[0]], meta.iloc[0])
    print(json.dumps(s, indent=1, ensure_ascii=False, default=str)[:2000])


if __name__ == "__main__":
    main()
