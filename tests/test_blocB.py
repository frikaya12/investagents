"""Tests des fonctions pures du bloc B."""

import numpy as np
import pytest

from investagents.ml.run_blocB import logit, make_emb_model, stack_features


def test_logit_is_bounded_and_symmetric():
    assert logit([0.5])[0] == pytest.approx(0.0)
    assert np.isfinite(logit([0.0, 1.0])).all()  # écrêtage, pas d'infini
    assert logit([0.8])[0] == pytest.approx(-logit([0.2])[0])


def test_stack_features_shape():
    x = stack_features([0.2, 0.3], [0.4, 0.5])
    assert x.shape == (2, 2)


def test_emb_model_learns_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 5))
    y = (x[:, 0] > 0).astype(int)
    m = make_emb_model(1.0).fit(x, y)
    assert (m.predict(x) == y).mean() > 0.95
