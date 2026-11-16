"""Calibration + backtest de portefeuille (MVP), sans LLM.

1. M1 et M1-texte (hyperparamètres figés en phase 2) sont entraînés sur l'ENTRAÎNEMENT (≤ 2007),
   prédits sur la validation et le test. Recalibration de Platt apprise sur la VALIDATION
   (2008–2011), appliquée au test. Brier et ECE avant / après, sur le test.
2. Backtest sur le test (2012–2017) : pour chaque cohorte (exercice d'approbation), budget =
   10 %, 25 % ou 50 % du montant total demandé par la cohorte ; sac à dos CP-SAT (max 25 % du
   budget par pays). Mesure : part de projets satisfaisants dans les portefeuilles, toutes
   cohortes réunies, IC 95 % de Wilson ; comparée à M1, au hasard (1 000 portefeuilles aléatoires
   sous les mêmes contraintes) et au taux de base de la cohorte.
Aucun réglage sur le test : budgets, plafond pays et modèles fixés avant.

Projets du backtest : test avec texte (pour comparer M1-texte et M1 sur les mêmes projets),
montant à l'approbation connu et pays connu.

Lancer : `uv run python -m investagents.selection.run_backtest`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from investagents.data.feasibility import wilson_ci
from investagents.ml import models as md
from investagents.ml.calibration import PlattCalibrator, ece
from investagents.ml.features import build_features
from investagents.ml.run_phase2 import FROZEN_PATH, load_data, load_texts
from investagents.selection.knapsack import random_portfolio, select_portfolio

OUT_DIR = Path("experiments/mvp")
REPORT = Path("reports/backtest_mvp.md")
BUDGET_FRACTIONS = [0.10, 0.25, 0.50]
MAX_COUNTRY_SHARE = 0.25
N_RANDOM = 1000


def fit_predict(table, y, split):
    """Prédictions d'échec (non calibrées) de M1 et M1-texte, modèles entraînés sur train."""
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    tr, va, te = split == "train", split == "valid", split == "test"
    x = build_features(table, tuple(frozen["m1"]["groups"]))
    m1 = md.make_m1(**frozen["m1"]["params"]).fit(x[tr], y[tr])
    out = pd.DataFrame(index=y.index)
    out["split"] = split
    out["y_fail"] = y
    out["m1"] = m1.predict_proba(x)[:, 1]

    texts = load_texts(y.index[tr | va | te])
    has = pd.Series(y.index.isin(texts.index), index=y.index)
    mt = md.make_m1_text(**frozen["m1_text"]["params"])
    mt.fit(texts.loc[y.index[tr & has]].tolist(), y[tr & has])
    sel = (va | te) & has
    out["m1_text"] = np.nan
    out.loc[sel, "m1_text"] = mt.predict_proba(texts.loc[y.index[sel]].tolist())[:, 1]
    return out


def calibrate(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    for m in ["m1", "m1_text"]:
        va = pred[(pred.split == "valid") & pred[m].notna()]
        te = pred[(pred.split == "test") & pred[m].notna()]
        cal = PlattCalibrator().fit(va.y_fail, va[m])
        pred.loc[pred[m].notna(), f"{m}_cal"] = cal.transform(pred.loc[pred[m].notna(), m])
        report[m] = {
            "n_valid": len(va),
            "n_test": len(te),
            "test_brier_before": float(brier_score_loss(te.y_fail, te[m])),
            "test_brier_after": float(brier_score_loss(te.y_fail, te[f"{m}_cal"])),
            "test_ece_before": ece(te.y_fail, te[m]),
            "test_ece_after": ece(te.y_fail, te[f"{m}_cal"]),
            "test_mean_pred_before": float(te[m].mean()),
            "test_mean_pred_after": float(te[f"{m}_cal"].mean()),
            "test_failure_rate": float(te.y_fail.mean()),
        }
    return pred, report


def backtest(pool: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(0)
    results, chosen_rows = {}, []
    for f in BUDGET_FRACTIONS:
        sel = {"m1_text": [], "m1": []}
        rand_success = np.zeros(N_RANDOM)
        rand_count = np.zeros(N_RANDOM)
        for _fy, c in pool.groupby("approval_fy"):
            budget = f * c.cost.sum()
            for m in sel:
                x = select_portfolio(
                    1 - c[f"{m}_cal"], c.cost, c.country, budget, MAX_COUNTRY_SHARE
                )
                sel[m].append(c[x])
                if m == "m1_text":
                    chosen_rows.append(c[x].assign(budget_fraction=f))
            for r in range(N_RANDOM):
                x = random_portfolio(c.cost, c.country, budget, MAX_COUNTRY_SHARE, rng)
                rand_success[r] += (1 - c.y_fail[x]).sum()
                rand_count[r] += x.sum()
        res = {"budget_fraction": f, "base_rate": float(1 - pool.y_fail.mean())}
        for m, parts in sel.items():
            s = pd.concat(parts)
            k, n = int((1 - s.y_fail).sum()), len(s)
            lo, hi = wilson_ci(k, n)
            res[m] = {
                "n_projects": n,
                "share_satisfactory": k / n,
                "ci_low": lo,
                "ci_high": hi,
                "spent_usd": float(s.cost.sum()),
            }
        share = rand_success / rand_count
        res["random"] = {
            "share_satisfactory": float(share.mean()),
            "p2_5": float(np.quantile(share, 0.025)),
            "p97_5": float(np.quantile(share, 0.975)),
            "mean_n_projects": float(rand_count.mean()),
        }
        # Part des tirages au hasard qui font au moins aussi bien que M1-texte (p-valeur empirique)
        res["p_value_vs_random"] = float(
            (1 + (share >= res["m1_text"]["share_satisfactory"]).sum()) / (N_RANDOM + 1)
        )
        results[f"{int(f * 100)}%"] = res
    return results, pd.concat(chosen_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, y, split = load_data()
    pred = fit_predict(table, y, split)
    pred, cal_report = calibrate(pred)

    meta = table.set_index("project_id")[
        ["approval_fy", "country", "amount_at_approval_usd", "global_practice", "outcome"]
    ]
    pred = pred.join(meta)
    pred.to_parquet(OUT_DIR / "predictions.parquet")

    pool = pred[
        (pred.split == "test")
        & pred.m1_text_cal.notna()
        & (pred.amount_at_approval_usd > 0)
        & pred.country.notna()
    ].rename(columns={"amount_at_approval_usd": "cost"})
    pool.to_parquet(OUT_DIR / "backtest_pool.parquet")
    results, chosen = backtest(pool)
    chosen.to_parquet(OUT_DIR / "portfolios_m1_text.parquet")
    out = {
        "calibration": cal_report,
        "backtest": results,
        "n_pool": len(pool),
        "budget_fractions": BUDGET_FRACTIONS,
        "max_country_share": MAX_COUNTRY_SHARE,
    }
    (OUT_DIR / "backtest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(out)


def write_report(out: dict) -> None:
    c = out["calibration"]
    lines = [
        "# Calibration et backtest de portefeuille (MVP, généré automatiquement)",
        "",
        "Généré par `uv run python -m investagents.selection.run_backtest`. Modèles entraînés sur",
        "≤ 2007, recalibrés (Platt) sur la validation 2008–2011, évalués sur le test 2012–2017.",
        "Aucun réglage sur le test.",
        "",
        "## Calibration (test, classe « échec »)",
        "",
        "| Modèle | Brier avant | Brier après | ECE avant | ECE après | Prédiction moyenne avant → après | Taux d'échec réel |",  # noqa: E501
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
        *[
            f"| {m} | {r['test_brier_before']:.4f} | {r['test_brier_after']:.4f} | "
            f"{r['test_ece_before']:.4f} | {r['test_ece_after']:.4f} | "
            f"{r['test_mean_pred_before']:.3f} → {r['test_mean_pred_after']:.3f} | "
            f"{r['test_failure_rate']:.3f} |"
            for m, r in c.items()
        ],
        "",
        f"## Backtest ({out['n_pool']} projets du test avec texte, montant et pays ; "
        f"plafond {out['max_country_share']:.0%} du budget par pays)",
        "",
        "Part de projets **satisfaisants** dans le portefeuille, toutes cohortes réunies.",
        "",
        "| Budget | M1-texte [IC 95 %] | M1 [IC 95 %] | Hasard (1 000 tirages) [2,5 % ; 97,5 %] | Taux de base | p (M1-texte vs hasard) |",  # noqa: E501
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for k, r in out["backtest"].items():
        mt, m1, rd = r["m1_text"], r["m1"], r["random"]
        lines.append(
            f"| {k} | {mt['share_satisfactory']:.1%} [{mt['ci_low']:.1%} ; {mt['ci_high']:.1%}] "
            f"(n = {mt['n_projects']}) | {m1['share_satisfactory']:.1%} [{m1['ci_low']:.1%} ; "
            f"{m1['ci_high']:.1%}] (n = {m1['n_projects']}) | {rd['share_satisfactory']:.1%} "
            f"[{rd['p2_5']:.1%} ; {rd['p97_5']:.1%}] | {r['base_rate']:.1%} | "
            f"{r['p_value_vs_random']:.3f} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
