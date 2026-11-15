"""Tests des métriques (bootstrap) et des modèles sur données synthétiques."""

import numpy as np
import pandas as pd
import pytest

from investagents.ml import metrics as mt
from investagents.ml import models as md


def test_metrics_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    perfect = mt.compute_metrics(y, [0.1, 0.2, 0.8, 0.9])
    assert perfect["auc_pr"] == pytest.approx(1.0)
    assert perfect["auc_roc"] == pytest.approx(1.0)
    constant = mt.compute_metrics(y, [0.5] * 4)
    assert constant["auc_roc"] == pytest.approx(0.5)
    assert constant["auc_pr"] == pytest.approx(0.5)  # = prévalence pour un score constant
    assert constant["brier"] == pytest.approx(0.25)


def test_bootstrap_ci_contains_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    a = mt.bootstrap_ci(y, p, n_boot=200, seed=1)
    b = mt.bootstrap_ci(y, p, n_boot=200, seed=1)
    assert a == b
    for m in ("auc_pr", "auc_roc", "brier"):
        assert a[m]["ci_low"] <= a[m]["value"] <= a[m]["ci_high"]


def test_bootstrap_diff_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    d = mt.bootstrap_diff(y, p, p, n_boot=100)
    assert d["auc_pr"]["value"] == 0
    assert d["auc_pr"]["ci_low"] == 0 and d["auc_pr"]["ci_high"] == 0


def test_bootstrap_diff_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y * 0.6 + rng.random(400) * 0.4
    bad = rng.random(400)
    d = mt.bootstrap_diff(y, good, bad, n_boot=200)
    assert d["auc_roc"]["ci_low"] > 0


def test_permutation_pvalue_signal_vs_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y * 0.5 + rng.random(300)
    noise = rng.random(300)
    assert mt.permutation_pvalue(y, good, n_perm=300)["p_value"] < 0.01
    assert mt.permutation_pvalue(y, noise, n_perm=300)["p_value"] > 0.05


def test_base_rate():
    m = md.BaseRate().fit(None, [0, 0, 0, 1])
    assert m.predict_proba([1, 2])[:, 1].tolist() == [0.25, 0.25]


def test_m1_learns_hidden_rule_on_synthetic_data():
    """Données synthétiques à règle cachée : échec si région A et petit montant."""
    rng = np.random.default_rng(0)
    n = 2000
    region = rng.choice(["A", "B", "C"], n)
    amount = rng.normal(17, 1, n)
    y = ((region == "A") & (amount < 17)).astype(int)
    x = pd.DataFrame({"region": pd.Categorical(region), "amount": amount})
    model = md.make_m1(max_iter=50).fit(x[:1500], y[:1500])
    score = mt.compute_metrics(y[1500:], model.predict_proba(x[1500:])[:, 1])
    assert score["auc_roc"] > 0.95


def test_m1_text_learns_word_signal():
    texts = ["risk delay weak capacity"] * 30 + ["strong ownership clear design"] * 30
    texts = [f"{t} filler document number {i}" for i, t in enumerate(texts)]
    y = np.array([1] * 30 + [0] * 30)
    model = md.make_m1_text(C=1.0)
    model.set_params(tfidf__min_df=1)
    model.fit(texts, y)
    p = model.predict_proba(["weak capacity delay", "clear strong ownership"])[:, 1]
    assert p[0] > p[1]


def test_tune_picks_best_on_validation():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.random(400)})
    y = (x["a"] > 0.5).astype(int).to_numpy()
    grid = {"max_iter": [1, 50], "learning_rate": [0.1]}
    best, log = md.tune(md.make_m1, grid, x[:300], y[:300], x[300:], y[300:])
    assert best["max_iter"] in (1, 50)
    assert log["valid_auc_pr"].iloc[0] == log["valid_auc_pr"].max()
