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
