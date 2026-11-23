"""M5 et M6 : facteurs de risque extraits par les agents (bloc D), seuls ou avec M1-texte.

CHOIX DE CODAGE (fixés après le pilote, sans regarder l'issue ; experiments/blocD/m6_choices.json,
commités avant tout réglage) :
- first_operation EXCLU : 1 seul « yes » juste sur 3 à la relecture du pilote ;
- chaque facteur = sa valeur + un indicateur « non trouvé » (1 si non trouvé, réponse incohérente
  ou citation invalide ; la valeur vaut alors 0) :
  - oui / non : 1 si « yes » ;
  - risques (low < moderate < substantial < high) : rang 0–3 ;
  - catégorie environnementale : indicatrices A, B, C, FI ;
  - listes (composantes, agences, cofinanceurs) : log(1 + min(nombre, 10)), nombre compté en
    Python.

Modèles (régression logistique, variables standardisées), réglés par validation croisée à 5 plis
sur les projets de VALIDATION ayant des facteurs (critère AUC-PR, grille de C) :
- M5 : facteurs seuls ;
- M6 : logit de M1-texte + facteurs.
Recalibration sans le test : Platt appris sur les prédictions hors pli de la validation, appliqué
aux prédictions du modèle réappris sur toute la validation. M1-texte : Platt sur la validation.
M0 : taux d'échec de la validation (constant).

Test UNE fois (`--stage evaluate` refuse de tourner deux fois) : écarts appariés M6 − M1-texte et
M5 − M0 (IC 95 % bootstrap), poids des facteurs, backtest du portefeuille (objectif figé
Σ p × coût, 25 % du budget, ≤ 25 % par pays, par cohorte) : M6 contre M1-texte contre 1 000
portefeuilles au hasard.

`--stage choices` (écrit les choix) ; `--stage select` ; `--stage evaluate`.
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
from investagents.selection.choose_objective import bootstrap_ci as portfolio_ci
from investagents.selection.choose_objective import (
    portfolio_metrics,
    random_draws,
    select_all_cohorts,
)

RUNS = Path("experiments/blocD")
SAMPLE = RUNS / "sample.csv"
PRED = Path("experiments/mvp/predictions.parquet")
CHOICES = RUNS / "m6_choices.json"
FROZEN = RUNS / "m6_frozen.json"
RESULTS_OUT = RUNS / "m6_results.json"
REPORT = Path("reports/blocD_m6.md")
EXCLUDED = ("first_operation",)
COUNT_CAP = 10
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
RISK_RANK = {"low": 0, "moderate": 1, "substantial": 2, "high": 3}
N_BOOT = 2000
SEED = 41
BUDGET = 0.25


def factor_row(factors: dict) -> dict:
    row = {}
    for f in FACTORS:
        if f.name in EXCLUDED:
            continue
        a = factors[f.name]
        ok = a["status"] == "found" and a["consistent"] and a.get("ids_valid", False)
        row[f"{f.name}_missing"] = int(not ok)
        if f.kind == "yes_no":
            row[f.name] = int(ok and a["value"] == "yes")
        elif f.kind == "list":
            n = int(a["value"]) if ok and a["value"] is not None else 0
            row[f.name] = float(np.log1p(min(n, COUNT_CAP)))
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
    df = df[df.m1_text.notna() & df.split.isin(["valid", "test"])].copy()
    df["logit_text"] = logit(df.m1_text)
    return df


def factor_columns(df: pd.DataFrame) -> list[str]:
    skip = {"key", "order", "split", "project_id", "y_fail", "approval_fy", "project_name",
            "doc_id", "n_candidates", "m1_text", "country", "amount_at_approval_usd",
            "logit_text"}  # fmt: skip
    return [c for c in df.columns if c not in skip]


def model_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    fc = factor_columns(df)
    return {"m5": fc, "m6": ["logit_text", *fc]}


def make(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def cv() -> StratifiedKFold:
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def stage_choices() -> None:
    if CHOICES.exists():
        raise SystemExit(f"Choix déjà figés : {CHOICES}")
    out = {
        "date": "2026-11-23",
        "basis": "pilote de 30 projets (validation), sans regarder l'issue",
        "excluded_factors": list(EXCLUDED),
        "excluded_reason": "first_operation : 1 « yes » juste sur 3 à la relecture du pilote",
        "coding": "valeur + indicateur « non trouvé » par facteur ; oui/non 0/1 ; risques rang "
        "0–3 ; catégorie environnementale indicatrices A/B/C/FI ; listes log(1 + min(n, 10))",
        "models": {"M5": "facteurs seuls", "M6": "logit M1-texte + facteurs"},
        "tuning": f"C ∈ {C_GRID}, VC 5 plis sur la validation, critère AUC-PR",
        "calibration": "Platt sur prédictions hors pli de la validation ; jamais le test",
        "backtest": f"objectif figé p_cost, budget {BUDGET:.0%}, ≤ 25 % par pays, par cohorte",
    }
    CHOICES.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Choix figés : {CHOICES}")


def stage_select() -> None:
    if not CHOICES.exists():
        raise SystemExit("Figer d'abord les choix (--stage choices).")
    df = load()
    va = df[df.split == "valid"]
    y = va.y_fail.to_numpy()
    out = {"n_valid": len(va), "valid_failure_rate": float(y.mean()), "models": {}}
    for name, cols in model_columns(df).items():
        grid = {}
        for c in C_GRID:
            p = cross_val_predict(make(c), va[cols], y, cv=cv(), method="predict_proba")[:, 1]
            grid[str(c)] = float(average_precision_score(y, p))
        best = max(grid, key=grid.get)
        out["models"][name] = {"C": float(best), "cv_auc_pr": grid[best], "columns": cols,
                               "grid": grid}  # fmt: skip
    ref = cross_val_predict(make(1.0), va[["logit_text"]], y, cv=cv(), method="predict_proba")
    out["cv_auc_pr_text_alone"] = float(average_precision_score(y, ref[:, 1]))
    FROZEN.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("C", "cv_auc_pr")}
                      if k == "models" else v for k, v in out.items()}, indent=2))  # fmt: skip


def fit_calibrated(va, te, cols, c):
    """Modèle appris sur la validation ; Platt sur ses prédictions hors pli (sans le test)."""
    yv = va.y_fail.to_numpy()
    oof = cross_val_predict(make(c), va[cols], yv, cv=cv(), method="predict_proba")[:, 1]
    model = make(c).fit(va[cols], yv)
    cal = PlattCalibrator().fit(yv, oof)
    return model, cal.transform(model.predict_proba(te[cols])[:, 1])


def backtest(te: pd.DataFrame, probs: dict) -> list[dict]:
    pool = te[(te.amount_at_approval_usd > 0) & te.country.notna()].rename(
        columns={"amount_at_approval_usd": "cost"}
    )
    pool = pool.assign(success=1 - pool.y_fail)
    rand = random_draws(pool, BUDGET, 1000, SEED)
    rows = []
    for name, p in probs.items():
        chosen = select_all_cohorts(pool.assign(p_success=1 - p[pool.index]), BUDGET,
                                    "p_cost", None)  # fmt: skip
        m = {**portfolio_metrics(chosen), **portfolio_ci(chosen)}
        m["p_vs_random_count"] = float(
            (rand.count_share_satisfactory >= m["count_share_satisfactory"]).mean()
        )
        m["p_vs_random_budget"] = float(
            (rand.budget_share_satisfactory >= m["budget_share_satisfactory"]).mean()
        )
        rows.append({"model": name, **m})
    rows.append({"model": "hasard", **rand.mean().to_dict(), "n_pool": len(pool)})
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
    preds = {"m0": np.full(len(te), yv.mean())}
    preds["texte"] = PlattCalibrator().fit(yv, va.m1_text.to_numpy()).transform(
        te.m1_text.to_numpy()
    )
    weights = {}
    for name in ("m5", "m6"):
        m = fr["models"][name]
        model, preds[name] = fit_calibrated(va, te, m["columns"], m["C"])
        weights[name] = dict(zip(m["columns"], model[-1].coef_[0].round(4).tolist(), strict=True))
    res = {"n_test": len(te), "test_failure_rate": float(yt.mean()), "models": {}}
    for name, p in preds.items():
        res["models"][name] = {
            "ci": bootstrap_ci(yt, p, N_BOOT, seed=SEED) if name != "m0" else None,
            "metrics": compute_metrics(yt, p) if name != "m0" else None,
            "brier": float(np.mean((p - yt) ** 2)),
            "ece": ece(yt, p),
        }
    res["diffs"] = {
        "m6 - texte": bootstrap_diff(yt, preds["m6"], preds["texte"], N_BOOT, SEED),
        "m5 - m0": bootstrap_diff(yt, preds["m5"], preds["m0"], N_BOOT, SEED),
    }
    res["weights"] = weights
    res["backtest"] = backtest(te, {k: pd.Series(preds[k], index=te.index)
                                    for k in ("m6", "texte")})  # fmt: skip
    RESULTS_OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame({"key": te.key, "y_fail": yt, **{f"p_{k}": v for k, v in preds.items()}}
                 ).to_parquet(RUNS / "m6_test_predictions.parquet", index=False)  # fmt: skip
    write_report(fr, res)


def fmt(d) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f} ; {d['ci_high']:.3f}]"


def write_report(fr: dict, res: dict) -> None:
    lines = [
        "# Bloc D : M5 et M6 — facteurs extraits par les agents (généré)",
        "",
        "Choix de codage figés avant réglage : experiments/blocD/m6_choices.json.",
        f"Réglage : VC à 5 plis dans les {fr['n_valid']} projets de validation (taux d'échec "
        f"{fr['valid_failure_rate']:.1%}). AUC-PR (VC) : M5 {fr['models']['m5']['cv_auc_pr']:.3f}"
        f" (C = {fr['models']['m5']['C']}), M6 {fr['models']['m6']['cv_auc_pr']:.3f}"
        f" (C = {fr['models']['m6']['C']}), texte seul {fr['cv_auc_pr_text_alone']:.3f}.",
        f"**Test : {res['n_test']} projets (taux d'échec {res['test_failure_rate']:.1%}), "
        "calculé une seule fois.** Probabilités recalibrées sans le test.",
        "",
        "| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    labels = {"m0": "M0 (taux de base)", "texte": "M1-texte (Platt)", "m5": "M5 (facteurs seuls)",
              "m6": "M6 (texte + facteurs)"}  # fmt: skip
    for k, m in res["models"].items():
        if k == "m0":
            lines.append(f"| {labels[k]} | {res['test_failure_rate']:.3f} | 0.500 | "
                         f"{m['brier']:.4f} | {m['ece']:.3f} |")  # fmt: skip
        else:
            lines.append(f"| {labels[k]} | {fmt(m['ci']['auc_pr'])} | {fmt(m['ci']['auc_roc'])} "
                         f"| {m['brier']:.4f} | {m['ece']:.3f} |")  # fmt: skip
    lines += ["", "| Écart apparié | AUC-PR | AUC-ROC | Brier |", "| --- | --- | --- | --- |"]
    for k, d in res["diffs"].items():
        lines.append(f"| {k} | {fmt(d['auc_pr'])} | {fmt(d['auc_roc'])} | {fmt(d['brier'])} |")
    lines += [
        "",
        f"## Backtest (objectif figé Σ p × coût, budget {BUDGET:.0%}, ≤ 25 % par pays)",
        "",
        "| Probabilités | projets | % réussis [IC 95 %] | p vs hasard | % budget → réussis "
        "[IC 95 %] | p vs hasard |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    for r in res["backtest"]:
        if r["model"] == "hasard":
            lines.append(f"| hasard (1 000 tirages) | {r['n_projects']:.0f} | "
                         f"{r['count_share_satisfactory']:.1%} | — | "
                         f"{r['budget_share_satisfactory']:.1%} | — |")  # fmt: skip
            continue
        c, b = r["count_share_ci"], r["budget_share_ci"]
        lines.append(
            f"| {r['model']} | {r['n_projects']} | {r['count_share_satisfactory']:.1%} "
            f"[{c[0]:.1%} ; {c[1]:.1%}] | {r['p_vs_random_count']:.3f} | "
            f"{r['budget_share_satisfactory']:.1%} [{b[0]:.1%} ; {b[1]:.1%}] | "
            f"{r['p_vs_random_budget']:.3f} |"
        )
    lines += ["", "## Poids (coefficients standardisés ; positif = plus d'échec)", ""]
    for name, w in res["weights"].items():
        top = sorted(w.items(), key=lambda kv: -abs(kv[1]))
        lines.append(f"- {name} : " + ", ".join(f"{k} {v:+.3f}" for k, v in top))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["choices", "select", "evaluate"], required=True)
    {"choices": stage_choices, "select": stage_select,
     "evaluate": stage_evaluate}[parser.parse_args().stage]()  # fmt: skip


if __name__ == "__main__":
    main()
