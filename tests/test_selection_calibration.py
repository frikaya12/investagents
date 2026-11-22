"""Tests : sac à dos CP-SAT, portefeuille au hasard, ECE et Platt."""

import numpy as np
import pytest

from investagents.ml.calibration import PlattCalibrator, ece
from investagents.selection.knapsack import random_portfolio, select_portfolio


def test_knapsack_respects_budget_and_picks_best():
    p = [0.9, 0.8, 0.1, 0.7]
    cost = [50e6, 50e6, 10e6, 60e6]
    country = ["A", "B", "C", "D"]
    x = select_portfolio(p, cost, country, budget_usd=100e6, max_country_share=1.0)
    assert x.tolist() == [True, True, False, False]  # 0,9 + 0,8 dans 100 M
    assert np.dot(x, cost) <= 100e6


def test_knapsack_country_cap():
    p = [0.9, 0.9, 0.5]
    cost = [30e6, 30e6, 30e6]
    country = ["A", "A", "B"]
    x = select_portfolio(p, cost, country, budget_usd=100e6, max_country_share=0.4)
    # Un seul des deux projets (identiques) du pays A, car 40 M max par pays ; + le projet B.
    assert x[:2].sum() == 1 and x[2]


def test_random_portfolio_respects_constraints():
    rng = np.random.default_rng(0)
    cost = rng.uniform(1e6, 50e6, 200)
    country = rng.choice(list("ABCDE"), 200)
    for _ in range(20):
        x = random_portfolio(cost, country, 300e6, 0.25, rng)
        assert cost[x].sum() <= 300e6
        for k in "ABCDE":
            assert cost[x & (country == k)].sum() <= 0.25 * 300e6 + 1e-6


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1] * 50)
    assert ece(y, np.full(200, 0.5)) == pytest.approx(0.0)
    assert ece(y, np.full(200, 0.9)) == pytest.approx(0.4)


def test_platt_corrects_shift():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.4, 3000)
    y = (rng.random(3000) < p_true).astype(int)
    p_biased = np.clip(p_true + 0.2, 0, 1)  # modèle trop pessimiste
    cal = PlattCalibrator().fit(y[:1500], p_biased[:1500])
    after = ece(y[1500:], cal.transform(p_biased[1500:]))
    assert after < ece(y[1500:], p_biased[1500:]) / 2


def test_calibrate_report_on_fake_predictions():
    import pandas as pd

    from investagents.selection.run_backtest import calibrate

    rng = np.random.default_rng(0)
    n = 400
    p = rng.uniform(0.1, 0.5, n)
    pred = pd.DataFrame(
        {
            "split": ["valid"] * 200 + ["test"] * 200,
            "y_fail": (rng.random(n) < p - 0.05).astype(int),
            "m1": p,
            "m1_text": np.where(np.arange(n) % 3 == 0, np.nan, p),
        }
    )
    out, report = calibrate(pred)
    assert out["m1_cal"].notna().all()
    assert out["m1_text_cal"].isna().sum() == pred["m1_text"].isna().sum()
    assert report["m1"]["test_ece_after"] <= report["m1"]["test_ece_before"]


def test_objective_p_prefers_many_cheap_projects():
    # Un gros projet très sûr contre deux petits moins sûrs : "p" compte les projets.
    p, cost, country = [0.95, 0.6, 0.6], [100e6, 50e6, 50e6], ["A", "B", "C"]
    x = select_portfolio(p, cost, country, 100e6, max_country_share=1.0, objective="p")
    assert x.tolist() == [False, True, True]


def test_objective_p_cost_prefers_high_probability_per_dollar():
    p, cost, country = [0.95, 0.6, 0.6], [100e6, 50e6, 50e6], ["A", "B", "C"]
    x = select_portfolio(p, cost, country, 100e6, max_country_share=1.0, objective="p_cost")
    assert x.tolist() == [True, False, False]


def test_objective_p_cost_min_excludes_low_probability():
    p, cost, country = [0.95, 0.5, 0.9], [60e6, 30e6, 30e6], ["A", "B", "C"]
    x = select_portfolio(
        p, cost, country, 100e6, max_country_share=1.0, objective="p_cost_min", p_min=0.7
    )
    assert not x[1] and x[0] and x[2]
    with pytest.raises(ValueError):
        select_portfolio(p, cost, country, 100e6, objective="p_cost_min")
    with pytest.raises(ValueError):
        select_portfolio(p, cost, country, 100e6, objective="unknown")


def test_weighted_score_bounds_and_weights():
    import pandas as pd
    import pytest

    from investagents.schemas.committee import CRITERIA
    from investagents.selection.multicriteria import weighted_score

    df = pd.DataFrame([{c: 10 for c in CRITERIA}, {c: 0 for c in CRITERIA}])
    s = weighted_score(df, {c: 1 for c in CRITERIA})
    assert s.tolist() == [1.0, 0.0]
    df2 = pd.DataFrame([{**{c: 0 for c in CRITERIA}, "risk": 10}])
    assert weighted_score(df2, {"risk": 1})[0] == 1.0  # seul le critère pondéré compte
    with pytest.raises(ValueError):
        weighted_score(df, {c: 0 for c in CRITERIA})


def _fake_pool(n=40, seed=0):
    import numpy as np
    import pandas as pd

    from investagents.schemas.committee import CRITERIA

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(2, 9, n) for c in CRITERIA})
    return df.assign(
        key=[f"K{i}" for i in range(n)],
        p_success=rng.uniform(0.5, 0.95, n),
        cost=rng.uniform(1e6, 1e8, n),
        country=rng.choice(list("ABCDEFGH"), n),
        approval_fy=rng.choice([2009, 2010], n),
        success=rng.integers(0, 2, n),
    )


def test_combined_score_extremes():
    import pytest

    from investagents.selection.frontier import combined_score, committee_score

    df = _fake_pool()
    assert (combined_score(df, 1.0) == df.p_success).all()  # α = 1 : ML seul
    assert (combined_score(df, 0.0) == committee_score(df)).all()  # α = 0 : comité seul
    mid = combined_score(df, 0.5)
    assert ((mid >= 0) & (mid <= 1)).all()
    with pytest.raises(ValueError):
        combined_score(df, 1.5)


def test_frontier_select_respects_cohort_budget_and_country_cap():
    from investagents.selection.frontier import combined_score, select

    df = _fake_pool()
    chosen = select(df, combined_score(df, 0.5), budget=0.25)
    for fy, c in df.groupby("approval_fy"):
        ch = chosen[chosen.approval_fy == fy]
        budget = 0.25 * c.cost.sum()
        assert ch.cost.sum() <= budget + 1e3 * len(ch)  # arrondi au millier de USD
        assert (ch.groupby("country").cost.sum() <= 0.25 * budget + 1e3 * len(ch)).all()
