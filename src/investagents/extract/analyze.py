"""Bloc D : contrôle du pilote / du passage complet de l'extracteur (sans regarder l'issue).

Critères du pilote FIXÉS À L'AVANCE (demande d'Aya) :
- citations valides ≥ 95 % (réponses « found » cohérentes dont tous les numéros cités existent
  parmi les phrases montrées) ;
- temps médian ≤ 40 s par projet (un GPU) ;
- la plupart des facteurs (≥ 10 sur 19) trouvés dans plus de 30 % des dossiers.
Contrôle complémentaire (diagnostic) : part des citations dont une phrase contient un mot-clé du
facteur.

Lancer : `uv run python -m investagents.extract.analyze --run experiments/blocD/kaggle_pilot`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from investagents.committee_run.runner import load_jsonl
from investagents.extract.factors import FACTORS

CRIT_CITATIONS = 0.95
CRIT_SECONDS = 40.0
CRIT_FOUND_RATE = 0.30
CRIT_N_FACTORS = 10
# Passage complet comparé au pilote (seuils fixés AVANT de voir le passage complet) :
MAX_ERROR_RATE = 0.02
MAX_NOT_FOUND_SHIFT = 0.25  # écart absolu de part « non trouvé » par facteur (pilote : n = 30)


def compare(summary: dict, table: pd.DataFrame, pilot_table: pd.DataFrame) -> dict:
    """« Nettement pire que le pilote » si l'un des seuils est franchi."""
    m = table.merge(pilot_table, on="factor", suffixes=("", "_pilot"))
    m["shift"] = (m.found_rate - m.found_rate_pilot).abs()
    worse = {
        "error_rate>2%": summary["n_errors"] / max(summary["n_projects"], 1) > MAX_ERROR_RATE,
        "citations<95%": not summary["criteria"]["citations>=95%"],
        "median>40s": not summary["criteria"]["median<=40s"],
        "<10 factors >30%": not summary["criteria"][">=10 factors found in >30%"],
        "factors shifted >25 pts": m.loc[m["shift"] > MAX_NOT_FOUND_SHIFT, "factor"].tolist(),
    }
    return {"much_worse": any(bool(v) for v in worse.values()), "checks": worse,
            "shift_by_factor": dict(zip(m.factor, m["shift"].round(3), strict=True))}  # fmt: skip


def summarize(results: list[dict]) -> tuple[dict, pd.DataFrame]:
    ok = [r for r in results if r.get("factors")]
    rows = []
    for f in FACTORS:
        fac = [r["factors"][f.name] for r in ok]
        found = [a for a in fac if a["status"] == "found" and a["consistent"]]
        rows.append({
            "factor": f.name,
            "kind": f.kind,
            "found_rate": len(found) / len(fac) if fac else 0.0,
            "inconsistent": sum(not a["consistent"] for a in fac),
            "ids_valid_rate": sum(a["ids_valid"] for a in found) / len(found) if found else None,
            "keyword_rate": sum(a["keyword_in_cited"] for a in found) / len(found)
            if found else None,
            "values": pd.Series([a["value"] for a in found]).value_counts().head(4).to_dict()
            if found else {},
        })  # fmt: skip
    table = pd.DataFrame(rows)
    answers = [a for r in ok for a in r["factors"].values()
               if a["status"] == "found" and a["consistent"]]  # fmt: skip
    seconds = pd.Series([r["seconds"] for r in ok])
    n_good = int((table.found_rate > CRIT_FOUND_RATE).sum())
    cit = sum(a["ids_valid"] for a in answers) / len(answers) if answers else 0.0
    summary = {
        "n_projects": len(results),
        "n_ok": len(ok),
        "n_errors": len(results) - len(ok),
        "citation_valid_rate": cit,
        "keyword_rate": sum(a["keyword_in_cited"] for a in answers) / len(answers)
        if answers else 0.0,
        "median_seconds": float(seconds.median()) if len(seconds) else None,
        "p90_seconds": float(seconds.quantile(0.9)) if len(seconds) else None,
        "n_factors_found_over_30pct": n_good,
        "criteria": {
            "citations>=95%": bool(cit >= CRIT_CITATIONS),
            "median<=40s": bool(len(seconds) and seconds.median() <= CRIT_SECONDS),
            ">=10 factors found in >30%": bool(n_good >= CRIT_N_FACTORS),
        },
    }  # fmt: skip
    summary["pilot_ok"] = bool(all(summary["criteria"].values()) and summary["n_errors"] == 0)
    return summary, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", default="reports/blocD_pilote.md")
    parser.add_argument("--pilot", default=None, help="dossier du pilote, pour comparer")
    args = parser.parse_args()
    results = load_jsonl(Path(args.run) / "results.jsonl")
    results = list({r["key"]: r for r in results}.values())
    summary, table = summarize(results)
    lines = [
        f"# Bloc D : extracteur — contrôle ({args.run}) (généré)",
        "",
        f"- Projets : {summary['n_ok']} réussis / {summary['n_projects']} ; erreurs "
        f"{summary['n_errors']}.",
        f"- Citations valides : {summary['citation_valid_rate']:.1%} (critère ≥ 95 %) ; phrase "
        f"citée contenant un mot-clé du facteur : {summary['keyword_rate']:.1%}.",
        f"- Temps par projet (un GPU) : médiane {summary['median_seconds']:.0f} s, 90e centile "
        f"{summary['p90_seconds']:.0f} s (critère ≤ 40 s).",
        f"- Facteurs trouvés dans plus de 30 % des dossiers : "
        f"{summary['n_factors_found_over_30pct']} / {len(FACTORS)} (critère ≥ 10).",
        f"- **Pilote bon : {'oui' if summary['pilot_ok'] else 'NON'}** {summary['criteria']}",
        "",
        "| Facteur | type | trouvé | incohérent | citations valides | mot-clé | valeurs |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in table.itertuples():
        iv = f"{r.ids_valid_rate:.0%}" if r.ids_valid_rate is not None else "—"
        kw = f"{r.keyword_rate:.0%}" if r.keyword_rate is not None else "—"
        lines.append(f"| {r.factor} | {r.kind} | {r.found_rate:.0%} | {r.inconsistent} | {iv} | "
                     f"{kw} | {r.values} |")  # fmt: skip
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.run, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
