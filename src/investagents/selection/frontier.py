"""Décision multicritère ML + agents : frontière en α (bloc C+, point 1).

Score d'un projet :
    score = α · p_succès + (1 − α) · moyenne pondérée des 5 notes du comité / 10
- p_succès : probabilité de réussite de M1-texte (entraîné ≤ 2007, recalibré par Platt sur la
  validation ; experiments/mvp/predictions.parquet, colonne m1_text_cal) ;
- notes : les 5 critères du Juge (inspirés des critères OCDE-CAD), poids égaux par défaut.

Sélection : même sac à dos que le portefeuille figé (selection/knapsack.py, objectif figé `p_cost`
du bloc A, où le score remplace p), une sélection par cohorte annuelle, budget = 25 % du montant de
la cohorte, au plus 25 % du budget par pays. Univers : les 600 projets notés par le comité
(experiments/blocC/scores.parquet), dont 8 sans montant à l'approbation sont exclus.

α ∈ {1 ; 0,75 ; 0,5 ; 0,25 ; 0} est FIXÉ À L'AVANCE ; aucun α n'est choisi. Pour chaque α :
part de projets réussis (et part du budget), taux de réussite attendu (moyenne de p_succès),
note moyenne du portefeuille sur chaque critère. Référence : 1 000 portefeuilles au hasard.

`--stage valid` : validation 2008–2011.
`--stage test`  : test 2012–2017, une seule fois, descriptif (refuse si déjà calculé).

Lancer : `uv run python -m investagents.selection.frontier --stage valid|test`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from investagents.schemas.committee import CRITERIA
from investagents.selection.choose_objective import bootstrap_ci, portfolio_metrics
from investagents.selection.knapsack import random_portfolio, select_portfolio

SCORES = Path("experiments/blocC/scores.parquet")
PRED = Path("experiments/mvp/predictions.parquet")
OUT = Path("experiments/blocC")
REPORT = Path("reports/frontiere_alpha.md")
ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
BUDGET = 0.25
MAX_COUNTRY_SHARE = 0.25
OBJECTIVE = "p_cost"  # objectif figé au bloc A
N_RANDOM = 1000
SEED = 7


def load_pool(split: str | None = None, scores_path=SCORES, pred_path=PRED) -> pd.DataFrame:
    """Projets notés par le comité, avec p_succès calibré, montant et pays."""
    df = pd.read_parquet(scores_path).dropna(subset=list(CRITERIA))
    pred = pd.read_parquet(pred_path)[["m1_text_cal"]]
    df = df.merge(pred, left_on="project_id", right_index=True, how="left")
    assert df.m1_text_cal.notna().all(), "projet sans probabilité calibrée"
    df = df[(df.amount_at_approval_usd > 0) & df.country.notna()]
    if split is not None:
        df = df[df.split == split]
    return df.assign(
        p_success=1 - df.m1_text_cal, success=1 - df.y_fail, cost=df.amount_at_approval_usd
    ).reset_index(drop=True)


def committee_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Moyenne pondérée des 5 notes, entre 0 et 1."""
    w = pd.Series({c: float((weights or {}).get(c, 1.0)) for c in CRITERIA})
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Les poids doivent être positifs et non tous nuls")
    return (df[list(CRITERIA)] * w).sum(axis=1) / (10 * w.sum())


def combined_score(df: pd.DataFrame, alpha: float, weights=None) -> pd.Series:
    if not 0 <= alpha <= 1:
        raise ValueError("α doit être entre 0 et 1")
    return alpha * df.p_success + (1 - alpha) * committee_score(df, weights)


def select(pool: pd.DataFrame, score: pd.Series, budget: float = BUDGET) -> pd.DataFrame:
    """Sac à dos par cohorte annuelle ; renvoie les projets financés."""
    parts = []
    for _, c in pool.assign(_score=score).groupby("approval_fy"):
        x = select_portfolio(
            c._score, c.cost, c.country, budget * c.cost.sum(), MAX_COUNTRY_SHARE,
            objective=OBJECTIVE,
        )  # fmt: skip
        parts.append(c[x])
    return pd.concat(parts).drop(columns="_score")


def profile(chosen: pd.DataFrame, with_outcome: bool = True) -> dict:
    """Profil d'un portefeuille : taille, réussite attendue, notes moyennes (± issue réelle)."""
    out = {
        "n_projects": int(len(chosen)),
        "expected_success": float(chosen.p_success.mean()),
        **{f"mean_{c}": float(chosen[c].mean()) for c in CRITERIA},
    }
    if with_outcome:
        out.update(portfolio_metrics(chosen))
    return out


def random_reference(pool: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    cohorts = [c for _, c in pool.groupby("approval_fy")]
    rows = []
    for _ in range(N_RANDOM):
        parts = [
            c[random_portfolio(c.cost, c.country, BUDGET * c.cost.sum(), MAX_COUNTRY_SHARE, rng)]
            for c in cohorts
        ]
        rows.append(profile(pd.concat(parts)))
    r = pd.DataFrame(rows)
    return {"mean": r.mean().to_dict(), "q025": r.quantile(0.025).to_dict(),
            "q975": r.quantile(0.975).to_dict()}  # fmt: skip


def frontier(pool: pd.DataFrame) -> dict:
    rows = []
    for a in ALPHAS:
        chosen = select(pool, combined_score(pool, a))
        ci = bootstrap_ci(chosen)
        rows.append({"alpha": a, **profile(chosen), "count_share_ci": ci["count_share_ci"],
                     "budget_share_ci": ci["budget_share_ci"],
                     "keys": sorted(chosen.key.tolist())})  # fmt: skip
    return {
        "n_pool": len(pool),
        "pool_success_rate": float(pool.success.mean()),
        "pool_means": {c: float(pool[c].mean()) for c in CRITERIA},
        "spread": {"p_success_std": float(pool.p_success.std()),
                   "committee_score_std": float(committee_score(pool).std())},  # fmt: skip
        "frontier": rows,
        "random": random_reference(pool),
    }


def run(split: str) -> dict:
    path = OUT / f"frontier_{split}.json"
    if split == "test":
        if not (OUT / "frontier_valid.json").exists():
            raise SystemExit("Validation d'abord (--stage valid).")
        if path.exists():
            raise SystemExit(f"Test déjà calculé ({path}) : une seule fois.")
    res = frontier(load_pool(split))
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


SHORT = {"relevance": "pert.", "economic_efficiency": "effic.", "feasibility": "faisab.",
         "risk": "risque", "impact_sustainability": "impact"}  # fmt: skip


def table(res: dict, title: str) -> list[str]:
    head = "| α | projets | réussis [IC 95 %] | budget → réussis | réussite attendue | "
    head += " | ".join(SHORT[c] for c in CRITERIA) + " |"
    lines = [
        f"### {title} ({res['n_pool']} projets, taux de réussite {res['pool_success_rate']:.1%})",
        "",
        head,
        "| " + " | ".join(["---"] + ["---:"] * (4 + len(CRITERIA))) + " |",
    ]
    for r in res["frontier"]:
        lo, hi = r["count_share_ci"]
        lines.append(
            f"| {r['alpha']:.2f} | {r['n_projects']} | {r['count_share_satisfactory']:.1%} "
            f"[{lo:.1%} ; {hi:.1%}] | {r['budget_share_satisfactory']:.1%} | "
            f"{r['expected_success']:.1%} | "
            + " | ".join(f"{r[f'mean_{c}']:.2f}" for c in CRITERIA) + " |"
        )
    m = res["random"]["mean"]
    lines.append(
        f"| hasard | {m['n_projects']:.0f} | {m['count_share_satisfactory']:.1%} | "
        f"{m['budget_share_satisfactory']:.1%} | {m['expected_success']:.1%} | "
        + " | ".join(f"{m[f'mean_{c}']:.2f}" for c in CRITERIA) + " |"
    )
    a1, a0 = res["frontier"][0], res["frontier"][-1]
    s = res["spread"]
    lines += [
        "",
        f"Écart-type dans l'univers : p_succès {s['p_success_std']:.3f}, score du comité "
        f"{s['committee_score_std']:.3f}.",
        f"De α = 1 à α = 0 : pertinence {a0['mean_relevance'] - a1['mean_relevance']:+.2f}, "
        f"impact {a0['mean_impact_sustainability'] - a1['mean_impact_sustainability']:+.2f} "
        f"(points sur 10) ; part de projets réussis "
        f"{100 * (a0['count_share_satisfactory'] - a1['count_share_satisfactory']):+.1f} points ; "
        f"réussite attendue {100 * (a0['expected_success'] - a1['expected_success']):+.1f} points.",
        "",
    ]
    return lines


def write_report() -> None:
    lines = [
        "# Frontière α : ML + agents dans la décision multicritère (généré)",
        "",
        "score = α · p_succès (M1-texte calibré) + (1 − α) · moyenne des 5 notes du comité / 10",
        f"(poids égaux). Sac à dos `{OBJECTIVE}` par cohorte annuelle, budget {BUDGET:.0%},",
        f"au plus {MAX_COUNTRY_SHARE:.0%} du budget par pays.",
        "α fixés à l'avance, aucun n'est choisi.",
        "Notes moyennes du portefeuille sur 10 ; « hasard » = moyenne de 1 000 portefeuilles.",
        "",
    ]
    valid = json.loads((OUT / "frontier_valid.json").read_text(encoding="utf-8"))
    lines += table(valid, "Validation 2008–2011")
    test_path = OUT / "frontier_test.json"
    if test_path.exists():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        lines += table(test, "Test 2012–2017 (une seule fois, descriptif)")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["valid", "test"], required=True)
    run(parser.parse_args().stage)
    write_report()


if __name__ == "__main__":
    main()
