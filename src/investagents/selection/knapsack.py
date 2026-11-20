"""Sélection de portefeuille sous budget : sac à dos résolu exactement avec OR-Tools (CP-SAT).

Objectifs disponibles (paramètre `objective`) :
- "p"          : maximiser Σ p_i · x_i (objectif initial du cahier des charges) ;
- "p_cost"     : maximiser Σ p_i · c_i · x_i (montant financé pondéré par la proba de succès) ;
- "p_cost_min" : comme "p_cost", mais seuls les projets avec p_i ≥ p_min sont éligibles.
(p_i = probabilité de succès calibrée du projet i, c_i = montant à l'approbation)
sous       Σ c_i · x_i ≤ B                       (budget)
           Σ_{i ∈ pays k} c_i · x_i ≤ α · B      (pas plus de α du budget par pays, α = 25 %)
           x_i ∈ {0, 1}
CP-SAT ne travaille qu'en entiers : coûts en milliers de USD, probabilités × 10 000.
"""

from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

COST_UNIT = 1_000  # USD -> milliers de USD
PROB_SCALE = 10_000


def select_portfolio(
    p_success,
    cost_usd,
    country,
    budget_usd: float,
    max_country_share: float = 0.25,
    time_limit_s: float = 10.0,
    objective: str = "p",
    p_min: float | None = None,
) -> np.ndarray:
    """Renvoie un masque booléen des projets financés."""
    if objective not in ("p", "p_cost", "p_cost_min"):
        raise ValueError(f"Objectif inconnu : {objective}")
    p = np.asarray(p_success, dtype=float)
    c = np.round(np.asarray(cost_usd, dtype=float) / COST_UNIT).astype(int)
    country = np.asarray(country)
    budget = int(budget_usd // COST_UNIT)
    cap = int(max_country_share * budget_usd // COST_UNIT)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(len(p))]
    model.Add(sum(int(c[i]) * x[i] for i in range(len(p))) <= budget)
    for k in np.unique(country):
        idx = np.flatnonzero(country == k)
        model.Add(sum(int(c[i]) * x[i] for i in idx) <= cap)
    if objective == "p_cost_min":
        if p_min is None:
            raise ValueError("p_min est requis pour l'objectif p_cost_min")
        for i in np.flatnonzero(p < p_min):
            model.Add(x[i] == 0)
    weight = [int(round(p[i] * PROB_SCALE)) for i in range(len(p))]
    if objective in ("p_cost", "p_cost_min"):
        weight = [weight[i] * int(c[i]) for i in range(len(p))]
    model.Maximize(sum(weight[i] * x[i] for i in range(len(p))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.random_seed = 0
    solver.parameters.num_workers = 1  # déterministe
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Aucune solution trouvée")
    return np.array([bool(solver.Value(v)) for v in x])


def random_portfolio(cost_usd, country, budget_usd, max_country_share, rng) -> np.ndarray:
    """Portefeuille au hasard : projets pris dans un ordre aléatoire tant que les contraintes
    (budget, part par pays) sont respectées. Référence « hasard »."""
    cost = np.asarray(cost_usd, dtype=float)
    country = np.asarray(country)
    chosen = np.zeros(len(cost), dtype=bool)
    spent, by_country = 0.0, {}
    for i in rng.permutation(len(cost)):
        k = country[i]
        if spent + cost[i] <= budget_usd and by_country.get(k, 0.0) + cost[i] <= (
            max_country_share * budget_usd
        ):
            chosen[i] = True
            spent += cost[i]
            by_country[k] = by_country.get(k, 0.0) + cost[i]
    return chosen
