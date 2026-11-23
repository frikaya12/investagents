"""Tests de fumée de la démo : chaque page s'exécute sans exception (sans navigateur), et les
chiffres affichés sont ceux des JSON de résultats (via report/metrics.py)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from investagents.report.metrics import load, num

APP = Path(__file__).resolve().parents[1] / "src" / "investagents" / "app" / "streamlit_app.py"
PAGES = ["accueil", "fiche", "resultats", "fonctionnement", "decision", "agents", "limites",
         "glossaire"]  # fmt: skip


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=240).run()
    if page != "accueil":
        at.switch_page(f"views/{page}.py").run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = run_page(page)
    assert not at.exception, at.exception
    assert at.title, "chaque page a un titre"


def test_home_numbers_come_from_results():
    t = load().txt
    at = run_page("accueil")
    shown = {mt.label: mt.value for mt in at.metric}
    assert t["pf25_share"] in shown.values()
    assert num(load().raw["m1t_aucpr"]["value"]) in shown.values()
    assert t["com_quotes_pct"] in shown.values()


def test_results_page_shows_vault():
    t = load().txt
    at = run_page("resultats")
    values = [mt.value for mt in at.metric]
    assert num(load().raw["vault_m1_aucroc"]["value"]) in values  # 0,509 bien visible
    assert t["ece_after"] in values


def test_fiche_reveal_button_shows_outcome():
    at = run_page("fiche")
    assert not at.error and not at.success  # issue cachée au départ
    at.button(key=[b.key for b in at.button if b.key and b.key.startswith("btn_")][0]).click().run()
    assert not at.exception
    assert at.error or at.success  # issue révélée


def test_decision_slider_updates_portfolio():
    at = run_page("decision")
    before = [mt.value for mt in at.metric]
    at.slider[0].set_value(0.0).run()
    assert not at.exception
    assert [mt.value for mt in at.metric] != before


def test_decision_matches_frozen_frontier():
    """À poids égaux et 25 % du budget, la démo retrouve les points de la frontière α figée."""
    import json

    from investagents.selection.frontier import combined_score, load_pool, select

    root = APP.parents[3]
    pool = load_pool("valid", root / "experiments/blocC/scores.parquet",
                     root / "experiments/mvp/predictions.parquet")  # fmt: skip
    fr = json.loads((root / "experiments/blocC/frontier_valid.json").read_text(encoding="utf-8"))
    for point in fr["frontier"]:
        chosen = select(pool, combined_score(pool, point["alpha"]), 0.25)
        assert len(chosen) == point["n_projects"]
        assert abs(chosen.success.mean() - point["count_share_satisfactory"]) < 1e-12


def test_project_sheet_content():
    from investagents.app.project_sheet import load_all, sheet

    recs, meta = load_all()
    row = meta.iloc[0]
    s = sheet(recs[row.key], row)
    assert 0 < s["p_success"] < 1
    assert len(s["scores"]) == 5 and len(s["arguments"]) == 10
    assert {a["stance"] for a in s["arguments"]} == {"pro", "contra"}
    assert all(a["cited"] for a in s["arguments"] if a["citation_valid"])


def test_random_draws_reproduce_frozen_results():
    """Les tirages de l'histogramme redonnent exactement la moyenne et les quantiles figés."""
    import json

    import numpy as np

    root = APP.parents[3]
    rd = json.loads((root / "experiments/demo/random_draws.json").read_text(encoding="utf-8"))
    frozen = next(r for r in json.loads((root / "experiments/blocA/objective_test.json")
                                        .read_text(encoding="utf-8"))["results"]
                  if r["budget"] == 0.25)["random"]["count_share_satisfactory"]  # fmt: skip
    d = np.array(rd["count_share"])
    assert abs(d.mean() - frozen["mean"]) < 1e-6
    assert abs(np.percentile(d, 2.5) - frozen["p2_5"]) < 1e-6


def test_calibration_ece_matches_results():
    import json

    import pandas as pd

    from investagents.ml.calibration import ece

    root = APP.parents[3]
    d = pd.read_parquet(root / "experiments/blocB/test_predictions.parquet")
    bb = json.loads((root / "experiments/blocB/results.json").read_text(encoding="utf-8"))
    assert abs(ece(d.y_fail, d.p_fail_m1_text) - bb["models"]["m1_text"]["ece_after"]) < 1e-12
