"""Bloc A.2 : choix de l'objectif de sélection sur la VALIDATION, puis test une seule fois.

Objectifs comparés (selection/knapsack.py) :
  (a) "p"          : Σ p_i            — objectif initial du cahier des charges
  (b) "p_cost"     : Σ p_i · c_i
  (c) "p_cost_min" : Σ p_i · c_i avec p_i ≥ taux de base de succès (mesuré sur la validation)
Probabilités : M1-texte entraîné sur ≤ 2007 et recalibré sur la validation (Platt), calculées par
selection/run_backtest.py (experiments/mvp/predictions.parquet).

Mesures, toutes cohortes (exercices d'approbation) réunies, pour 3 budgets (10, 25, 50 % du
montant demandé par cohorte), plafond 25 % du budget par pays :
- part du budget allant à des projets satisfaisants (pondérée par le montant) ;
- part des projets financés qui sont satisfaisants.

Critère de choix FIXÉ À L'AVANCE : moyenne, sur les 3 budgets, de la part du budget allant à des
projets satisfaisants, sur la validation. En cas d'égalité (< 0,1 point), l'objectif le plus simple.

`--stage select`   : validation seulement ; écrit experiments/blocA/objective_frozen.json.
`--stage evaluate` : test, une seule fois, objectif figé contre 1 000 portefeuilles au hasard ;
                     l'objectif initial (a) est montré à côté, pour transparence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from investagents.selection.knapsack import random_portfolio, select_portfolio

PRED_PATH = Path("experiments/mvp/predictions.parquet")
OUT_DIR = Path("experiments/blocA")
FROZEN = OUT_DIR / "objective_frozen.json"
REPORT = Path("reports/objectif_selection.md")
BUDGETS = [0.10, 0.25, 0.50]
MAX_COUNTRY_SHARE = 0.25
OBJECTIVES = ["p", "p_cost", "p_cost_min"]  # du plus simple au plus contraint
N_RANDOM_VALID = 200
N_RANDOM_TEST = 1000
N_BOOT = 2000


def load_pool(split: str) -> pd.DataFrame:
    pred = pd.read_parquet(PRED_PATH)
    pool = pred[
        (pred.split == split)
        & pred.m1_text_cal.notna()
        & (pred.amount_at_approval_usd > 0)
        & pred.country.notna()
    ].rename(columns={"amount_at_approval_usd": "cost"})
    pool = pool.assign(p_success=1 - pool.m1_text_cal, success=1 - pool.y_fail)
    return pool


def portfolio_metrics(chosen: pd.DataFrame) -> dict:
    return {
        "n_projects": int(len(chosen)),
        "spent_usd": float(chosen.cost.sum()),
        "budget_share_satisfactory": float(
            (chosen.cost * chosen.success).sum() / chosen.cost.sum()
        ),
        "count_share_satisfactory": float(chosen.success.mean()),
    }


def select_all_cohorts(pool, frac, objective, p_min) -> pd.DataFrame:
    parts = []
    for _, c in pool.groupby("approval_fy"):
        x = select_portfolio(
            c.p_success,
            c.cost,
            c.country,
            frac * c.cost.sum(),
            MAX_COUNTRY_SHARE,
            objective=objective,
            p_min=p_min,
        )
        parts.append(c[x])
    return pd.concat(parts)


def random_draws(pool, frac, n, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = [c for _, c in pool.groupby("approval_fy")]
    for _ in range(n):
        parts = [
            c[random_portfolio(c.cost, c.country, frac * c.cost.sum(), MAX_COUNTRY_SHARE, rng)]
            for c in cohorts
        ]
        rows.append(portfolio_metrics(pd.concat(parts)))
    return pd.DataFrame(rows)


def bootstrap_ci(chosen: pd.DataFrame, seed: int = 0) -> dict:
    """IC 95 % des deux parts, en rééchantillonnant les projets du portefeuille."""
    rng = np.random.default_rng(seed)
    c, s = chosen.cost.to_numpy(), chosen.success.to_numpy()
    budget, count = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(c), len(c))
        budget.append((c[i] * s[i]).sum() / c[i].sum())
        count.append(s[i].mean())
    q = [0.025, 0.975]
    return {
        "budget_share_ci": np.quantile(budget, q).tolist(),
        "count_share_ci": np.quantile(count, q).tolist(),
    }


def stage_select() -> None:
    pool = load_pool("valid")
    p_min = float(pool.success.mean())  # taux de base de succès sur la validation
    rows = []
    for obj in OBJECTIVES:
        for frac in BUDGETS:
            m = portfolio_metrics(select_all_cohorts(pool, frac, obj, p_min))
            rows.append({"objective": obj, "budget": frac, **m})
    for frac in BUDGETS:
        r = random_draws(pool, frac, N_RANDOM_VALID, seed=1).mean()
        rows.append({"objective": "hasard", "budget": frac, **r.to_dict()})
    table = pd.DataFrame(rows)
    score = (
        table[table.objective.isin(OBJECTIVES)]
        .groupby("objective")["budget_share_satisfactory"]
        .mean()
        .reindex(OBJECTIVES)
    )
    best = score.max()
    chosen = next(o for o in OBJECTIVES if score[o] >= best - 0.001)  # le plus simple à égalité
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "objective_validation.csv", index=False)
    FROZEN.write_text(
        json.dumps(
            {
                "objective": chosen,
                "p_min": p_min,
                "criterion": "moyenne budget_share_satisfactory",
                "validation_scores": score.to_dict(),
                "n_valid_pool": len(pool),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print("\nScore de choix (validation) :", score.round(4).to_dict(), "-> figé :", chosen)


def stage_evaluate() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    pool = load_pool("test")
    valid = pd.read_csv(OUT_DIR / "objective_validation.csv")
    results = []
    for frac in BUDGETS:
        rand = random_draws(pool, frac, N_RANDOM_TEST, seed=2)
        row = {
            "budget": frac,
            "base_rate_count": float(pool.success.mean()),
            "base_rate_budget": float((pool.cost * pool.success).sum() / pool.cost.sum()),
        }
        for label, obj in [("chosen", frozen["objective"]), ("initial_p", "p")]:
            chosen = select_all_cohorts(pool, frac, obj, frozen["p_min"])
            m = portfolio_metrics(chosen)
            row[label] = {
                **m,
                **bootstrap_ci(chosen),
                "objective": obj,
                "p_value_budget_vs_random": float(
                    (1 + (rand.budget_share_satisfactory >= m["budget_share_satisfactory"]).sum())
                    / (len(rand) + 1)
                ),
                "p_value_count_vs_random": float(
                    (1 + (rand.count_share_satisfactory >= m["count_share_satisfactory"]).sum())
                    / (len(rand) + 1)
                ),
            }
        row["random"] = {
            k: {
                "mean": float(rand[k].mean()),
                "p2_5": float(rand[k].quantile(0.025)),
                "p97_5": float(rand[k].quantile(0.975)),
            }
            for k in ["budget_share_satisfactory", "count_share_satisfactory"]
        }
        results.append(row)
    (OUT_DIR / "objective_test.json").write_text(
        json.dumps({"frozen": frozen, "results": results, "n_test_pool": len(pool)}, indent=2),
        encoding="utf-8",
    )
    write_report(frozen, valid, results, len(pool))


CONTROL_REPORT_MARKER = "## 3. Contrôle : d'où vient le gain ?"


def stage_control() -> None:
    """Contrôle (évaluation, pas réglage) : même objectif figé, autres probabilités, sur le test.

    - M0 : probabilité constante (taux de succès de l'entraînement + validation). Avec Σ p·c, seul
      le montant compte : le solveur remplit le budget au maximum, quelle que soit la qualité.
    - M1 : variables structurées, recalibré sur la validation.
    - M1-texte : rappel du résultat figé.
    Mêmes cohortes, budgets, plafond et tirages au hasard (graine 2) que l'évaluation figée.
    """
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    pred = pd.read_parquet(PRED_PATH)
    base_success = float(1 - pred[pred.split.isin(["train", "valid"])].y_fail.mean())
    pool = load_pool("test")
    variants = {
        "M0 (constante)": np.full(len(pool), base_success),
        "M1": 1 - pool.m1_cal.to_numpy(),
        "M1-texte": pool.p_success.to_numpy(),
    }
    results = []
    for frac in BUDGETS:
        rand = random_draws(pool, frac, N_RANDOM_TEST, seed=2)
        row = {
            "budget": frac,
            "random": {
                k: float(rand[k].mean())
                for k in ["budget_share_satisfactory", "count_share_satisfactory"]
            },
        }
        for name, p in variants.items():
            chosen = select_all_cohorts(pool.assign(p_success=p), frac, frozen["objective"], None)
            m = portfolio_metrics(chosen)
            row[name] = {
                **m,
                **bootstrap_ci(chosen),
                "p_value_budget_vs_random": float(
                    (1 + (rand.budget_share_satisfactory >= m["budget_share_satisfactory"]).sum())
                    / (len(rand) + 1)
                ),
                "p_value_count_vs_random": float(
                    (1 + (rand.count_share_satisfactory >= m["count_share_satisfactory"]).sum())
                    / (len(rand) + 1)
                ),
                "mean_cost_musd": float(chosen.cost.mean() / 1e6),
            }
        results.append(row)
    (OUT_DIR / "objective_control.json").write_text(
        json.dumps(
            {
                "objective": frozen["objective"],
                "base_success": base_success,
                "results": results,
                "pool_mean_cost_musd": float(pool.cost.mean() / 1e6),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        CONTROL_REPORT_MARKER,
        "",
        f"Même objectif figé (`{frozen['objective']}`), mêmes tirages au hasard, sur le test ;",
        "seules les probabilités changent. Avec M0 (probabilité constante), seul le montant",
        "des projets compte.",
        f"Montant moyen d'un projet du test : {pool.cost.mean() / 1e6:.0f} M USD.",
        "",
        "| Budget | Probabilités | % budget → satisfaisants [IC 95 %] | p vs hasard | % projets satisfaisants [IC 95 %] | p vs hasard | Projets | Montant moyen (M USD) |",  # noqa: E501
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in results:
        for name in variants:
            m = r[name]
            lines.append(
                f"| {r['budget']:.0%} | {name} | {pct(m['budget_share_satisfactory'])} "
                f"[{pct(m['budget_share_ci'][0])} ; {pct(m['budget_share_ci'][1])}] | "
                f"{m['p_value_budget_vs_random']:.3f} | {pct(m['count_share_satisfactory'])} "
                f"[{pct(m['count_share_ci'][0])} ; {pct(m['count_share_ci'][1])}] | "
                f"{m['p_value_count_vs_random']:.3f} | {m['n_projects']} | {m['mean_cost_musd']:.0f} |"  # noqa: E501
            )
        lines.append(
            f"| {r['budget']:.0%} | hasard (1 000) | {pct(r['random']['budget_share_satisfactory'])} "  # noqa: E501
            f"| — | {pct(r['random']['count_share_satisfactory'])} | — | — | — |"
        )
    text = REPORT.read_text(encoding="utf-8")
    if CONTROL_REPORT_MARKER in text:
        text = text[: text.index(CONTROL_REPORT_MARKER)].rstrip() + "\n"
    REPORT.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def pct(x):
    return f"{100 * x:.1f} %"


def write_report(frozen, valid, results, n_test) -> None:
    lines = [
        "# Choix de l'objectif de sélection (bloc A.2, généré automatiquement)",
        "",
        "Probabilités : M1-texte entraîné sur ≤ 2007, recalibré sur la validation. Plafond 25 % du",
        "budget par pays. Critère de choix fixé à l'avance : moyenne sur 3 budgets de la part du",
        "budget allant à des projets satisfaisants, **sur la validation**.",
        "",
        f"## 1. Validation 2008–2011 ({frozen['n_valid_pool']} projets) — sert au choix",
        "",
        "| Objectif | Budget | Projets financés | % du budget → projets satisfaisants | % de projets satisfaisants |",  # noqa: E501
        "| --- | --- | ---: | ---: | ---: |",
        *[
            f"| {r.objective} | {r.budget:.0%} | {r.n_projects:.0f} | "
            f"{pct(r.budget_share_satisfactory)} | {pct(r.count_share_satisfactory)} |"
            for r in valid.itertuples()
        ],
        "",
        "Score de choix : "
        + ", ".join(f"{k} = {pct(v)}" for k, v in frozen["validation_scores"].items())
        + f". **Objectif figé : `{frozen['objective']}`** (p minimum = {frozen['p_min']:.3f}).",
        "",
        f"## 2. Test 2012–2017 ({n_test} projets) — calculé une seule fois",
        "",
        "| Budget | Objectif | % budget → satisfaisants [IC 95 %] | p vs hasard | % projets satisfaisants [IC 95 %] | p vs hasard | Projets |",  # noqa: E501
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for r in results:
        for label, name in [
            ("chosen", f"**{r['chosen']['objective']} (figé)**"),
            ("initial_p", "p (initial, transparence)"),
        ]:
            m = r[label]
            lines.append(
                f"| {r['budget']:.0%} | {name} | {pct(m['budget_share_satisfactory'])} "
                f"[{pct(m['budget_share_ci'][0])} ; {pct(m['budget_share_ci'][1])}] | "
                f"{m['p_value_budget_vs_random']:.3f} | {pct(m['count_share_satisfactory'])} "
                f"[{pct(m['count_share_ci'][0])} ; {pct(m['count_share_ci'][1])}] | "
                f"{m['p_value_count_vs_random']:.3f} | {m['n_projects']} |"
            )
        rb, rc = r["random"]["budget_share_satisfactory"], r["random"]["count_share_satisfactory"]
        lines.append(
            f"| {r['budget']:.0%} | hasard (1 000 tirages) | {pct(rb['mean'])} "
            f"[{pct(rb['p2_5'])} ; {pct(rb['p97_5'])}] | — | {pct(rc['mean'])} "
            f"[{pct(rc['p2_5'])} ; {pct(rc['p97_5'])}] | — | — |"
        )
        lines.append(
            f"| {r['budget']:.0%} | taux de base de la cohorte | {pct(r['base_rate_budget'])} | — "
            f"| {pct(r['base_rate_count'])} | — | — |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "evaluate", "control"], required=True)
    args = parser.parse_args()
    {"select": stage_select, "evaluate": stage_evaluate, "control": stage_control}[args.stage]()


if __name__ == "__main__":
    main()
