"""M6 : M1-texte + facteurs de risque extraits par les agents (bloc D).

Variables (calculées en Python à partir des réponses vérifiées de l'extracteur) :
- oui / non : 1 si « yes » trouvé avec citation valide, sinon 0 ; plus un indicateur « trouvé » ;
- catégories de risque (low < moderate < substantial < high) : rang 0–3, 0 si non trouvé, plus
  l'indicateur « trouvé » ; catégorie environnementale : indicatrices A, B, C, FI ;
- listes (composantes, agences, cofinanceurs) : nombre d'éléments compté en Python (0 si non
  trouvé), plus l'indicateur « trouvé ».
Une réponse dont les numéros cités sont invalides est traitée comme « non trouvée ».

M6 = régression logistique sur [logit de M1-texte, variables des facteurs]. Apprentissage et
réglage (C, validation croisée à 5 plis) sur les projets de VALIDATION qui ont des facteurs ;
test UNE fois. Référence : M1-texte recalibré (Platt) sur les mêmes projets de validation.
Backtest : même sac à dos figé (p_cost, ≤ 25 % par pays, par cohorte), probabilités de M6
contre M1-texte et 1 000 portefeuilles au hasard, sur les projets du test qui ont des facteurs.

`--stage select` puis `--stage evaluate` (refuse de tourner deux fois).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from investagents.committee_run.runner import load_jsonl
from investagents.extract.factors import FACTORS
from investagents.ml.calibration import PlattCalibrator, ece
from investagents.ml.metrics import bootstrap_ci, bootstrap_diff, compute_metrics
from investagents.ml.run_blocC_models import logit
from investagents.selection.choose_objective import (
    bootstrap_ci as portfolio_ci,
)
from investagents.selection.choose_objective import (
    portfolio_metrics,
    random_draws,
    select_all_cohorts,
)

RUNS = Path("experiments/blocD")
SAMPLE = RUNS / "sample.csv"
PRED = Path("experiments/mvp/predictions.parquet")
FROZEN = RUNS / "m6_frozen.json"
RESULTS_OUT = RUNS / "m6_results.json"
REPORT = Path("reports/blocD_m6.md")
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
RISK_RANK = {"low": 0, "moderate": 1, "substantial": 2, "high": 3}
N_BOOT = 2000
SEED = 41
BUDGETS = [0.10, 0.25, 0.50]


def factor_row(factors: dict) -> dict:
    row = {}
    for f in FACTORS:
        a = factors[f.name]
        ok = a["status"] == "found" and a["consistent"] and a.get("ids_valid", False)
        row[f"{f.name}_found"] = int(ok)
        if f.kind == "yes_no":
            row[f.name] = int(ok and a["value"] == "yes")
        elif f.kind == "list":
            row[f.name] = int(a["value"]) if ok and a["value"] is not None else 0
        elif f.name == "env_category":
            for cat in f.categories:
                row[f"env_{cat}"] = int(ok and a["value"] == cat)
        else:
            row[f.name] = RISK_RANK.get(a["value"], 0) if ok else 0
    return row


def load() -> pd.DataFrame:
    recs = {}
    for path in sorted(RUNS.glob("kaggle_*/results.jsonl")):
        for r in load_jsonl(path):
            if r.get("factors"):
                recs[r["key"]] = r["factors"]
    feats = pd.DataFrame([{"key": k, **factor_row(v)} for k, v in recs.items()])
    sample = pd.read_csv(SAMPLE)
    pred = pd.read_parquet(PRED)[["m1_text", "country", "amount_at_approval_usd"]]
    df = sample.merge(feats, on="key").merge(pred, left_on="project_id", right_index=True)
    df = df[df.m1_text.notna()].copy()
    df["logit_text"] = logit(df.m1_text)
    return df


def columns(df: pd.DataFrame) -> list[str]:
    skip = {"key", "order", "split", "project_id", "y_fail", "approval_fy", "project_name",
            "doc_id", "n_candidates", "m1_text", "country", "amount_at_approval_usd"}  # fmt: skip
    return [c for c in df.columns if c not in skip]


def make(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def stage_select() -> None:
    df = load()
    va = df[df.split == "valid"]
    y = va.y_fail.to_numpy()
    cols = columns(df)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    grid = {}
    for c in C_GRID:
        p = cross_val_predict(make(c), va[cols], y, cv=cv, method="predict_proba")[:, 1]
        grid[str(c)] = float(average_precision_score(y, p))
    ref = cross_val_predict(make(1.0), va[["logit_text"]], y, cv=cv, method="predict_proba")
    best = max(grid, key=grid.get)
    out = {"n_valid": len(va), "valid_failure_rate": float(y.mean()), "columns": cols,
           "C": float(best), "cv_auc_pr": grid[best],
           "cv_auc_pr_text_alone": float(average_precision_score(y, ref[:, 1])),
           "grid": grid}  # fmt: skip
    FROZEN.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("grid", "columns")}, indent=2))


def backtest(te: pd.DataFrame, probs: dict) -> list[dict]:
    pool = te[(te.amount_at_approval_usd > 0) & te.country.notna()].rename(
        columns={"amount_at_approval_usd": "cost"}
    )
    pool = pool.assign(success=1 - pool.y_fail)
    rows = []
    for frac in BUDGETS:
        rand = random_draws(pool, frac, 1000, SEED)
        for name, p in probs.items():
            chosen = select_all_cohorts(pool.assign(p_success=1 - p[pool.index]), frac,
                                        "p_cost", None)  # fmt: skip
            m = portfolio_metrics(chosen)
            m.update(portfolio_ci(chosen))
            m["p_vs_random_count"] = float((rand.count_share_satisfactory
                                            >= m["count_share_satisfactory"]).mean())  # fmt: skip
            rows.append({"budget": frac, "model": name, **m})
        rows.append({"budget": frac, "model": "hasard",
                     **rand.mean().to_dict()})  # fmt: skip
    return rows


def stage_evaluate() -> None:
    if not FROZEN.exists():
        raise SystemExit("Lancer d'abord --stage select.")
    if RESULTS_OUT.exists():
        raise SystemExit(f"Déjà évalué ({RESULTS_OUT}) : le test ne se calcule qu'une fois.")
    fr = json.loads(FROZEN.read_text(encoding="utf-8"))
    df = load()
    va, te = df[df.split == "valid"], df[df.split == "test"]
    yv, yt = va.y_fail.to_numpy(), te.y_fail.to_numpy()
    p_text = PlattCalibrator().fit(yv, va.m1_text.to_numpy()).transform(te.m1_text.to_numpy())
    model = make(fr["C"]).fit(va[fr["columns"]], yv)
    p_m6 = model.predict_proba(te[fr["columns"]])[:, 1]
    res = {
        "n_test": len(te),
        "test_failure_rate": float(yt.mean()),
        "models": {
            name: {"metrics": compute_metrics(yt, p), "ci": bootstrap_ci(yt, p, N_BOOT, seed=SEED),
                   "brier": float(np.mean((p - yt) ** 2)), "ece": ece(yt, p)}  # fmt: skip
            for name, p in (("texte", p_text), ("m6", p_m6))
        },
        "diff_m6_texte": bootstrap_diff(yt, p_m6, p_text, N_BOOT, SEED),
        "coefs": dict(zip(fr["columns"], model[-1].coef_[0].round(4).tolist(), strict=True)),
        "backtest": backtest(te, {"m6": pd.Series(p_m6, index=te.index),
                                  "texte": pd.Series(p_text, index=te.index)}),  # fmt: skip
    }
    RESULTS_OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(fr, res)


def fmt(d) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f} ; {d['ci_high']:.3f}]"


def write_report(fr: dict, res: dict) -> None:
    lines = [
        "# Bloc D : M6 — M1-texte + facteurs extraits par les agents (généré)",
        "",
        f"Réglage : validation croisée à 5 plis dans les {fr['n_valid']} projets de validation "
        f"avec facteurs ; C = {fr['C']} ; AUC-PR (VC) {fr['cv_auc_pr']:.3f} contre "
        f"{fr['cv_auc_pr_text_alone']:.3f} pour le texte seul.",
        f"Test : {res['n_test']} projets (taux d'échec {res['test_failure_rate']:.1%}), une seule "
        "fois.",
        "",
        "| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for k, m in res["models"].items():
        lines.append(f"| {k} | {fmt(m['ci']['auc_pr'])} | {fmt(m['ci']['auc_roc'])} | "
                     f"{m['brier']:.4f} | {m['ece']:.3f} |")  # fmt: skip
    d = res["diff_m6_texte"]
    lines += [
        "",
        f"M6 − texte : AUC-PR {fmt(d['auc_pr'])}, AUC-ROC {fmt(d['auc_roc'])}, "
        f"Brier {fmt(d['brier'])}.",
        "",
        "## Backtest (sac à dos figé p_cost, ≤ 25 % par pays, par cohorte)",
        "",
        "| Budget | Modèle | projets | % réussis [IC 95 %] | p vs hasard | % budget → réussis |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for r in res["backtest"]:
        ci = r.get("count_share_ci")
        ci_s = f" [{ci[0]:.1%} ; {ci[1]:.1%}]" if ci else ""
        pv = f"{r['p_vs_random_count']:.3f}" if "p_vs_random_count" in r else "—"
        lines.append(f"| {r['budget']:.0%} | {r['model']} | {r['n_projects']:.0f} | "
                     f"{r['count_share_satisfactory']:.1%}{ci_s} | {pv} | "
                     f"{r['budget_share_satisfactory']:.1%} |")  # fmt: skip
    lines += ["", "Coefficients (standardisés, positif = plus d'échec) : "
              + ", ".join(f"{k} {v:+.2f}" for k, v in res["coefs"].items())]  # fmt: skip
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "evaluate"], required=True)
    stage_select() if parser.parse_args().stage == "select" else stage_evaluate()


if __name__ == "__main__":
    main()
