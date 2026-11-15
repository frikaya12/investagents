"""Tests de l'étude de faisabilité : intervalle de Wilson et classement des projets."""

import pandas as pd
import pytest

from investagents.data import feasibility as fe


def test_wilson_known_values():
    # Valeurs de référence : 10/50 -> [0,112 ; 0,330] (calcul manuel de la formule de Wilson)
    lo, hi = fe.wilson_ci(10, 50)
    assert lo == pytest.approx(0.1124, abs=1e-3)
    assert hi == pytest.approx(0.3304, abs=1e-3)


def test_wilson_edges_stay_in_0_1():
    lo, hi = fe.wilson_ci(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    lo, hi = fe.wilson_ci(50, 50)
    assert hi == 1.0 and 0.9 < lo < 1


def test_wilson_rejects_empty_sample():
    with pytest.raises(ValueError):
        fe.wilson_ci(0, 0)


def test_classify_projects_statuses():
    kept = pd.DataFrame(
        {
            "project_id": ["P1", "P2", "P2", "P3"],
            "tier": [1, 2, 1, 1],
            "n_chars": [50_000, 20_000, 100, 0],
        }
    )
    out = fe.classify_projects(["P1", "P2", "P3", "P4"], kept, min_chars=10_000)
    status = dict(zip(out["project_id"], out["status"], strict=True))
    assert status == {
        "P1": "A_tier1_text",
        "P2": "B_tier2_text",  # tier 1 trop court, tier 2 assez long
        "P3": "C_ex_ante_no_text",
        "P4": "D_no_ex_ante",
    }
    assert out.set_index("project_id").loc["P2", "max_chars"] == 20_000
