"""Test de fumée de la démo : l'application s'exécute sans exception (sans navigateur)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "investagents" / "app" / "streamlit_app.py"


def test_portfolio_page_runs():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception


def test_projects_page_runs():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio[0].set_value("Fiche projet").run()
    assert not at.exception


def test_multicriteria_page_runs():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Sélection multicritère (comité)").run()
    assert not at.exception


def test_project_sheet_content():
    from investagents.app.project_sheet import load_all, sheet

    recs, meta = load_all()
    row = meta.iloc[0]
    s = sheet(recs[row.key], row)
    assert 0 < s["p_success"] < 1
    assert len(s["scores"]) == 5 and len(s["arguments"]) == 10
    assert {a["stance"] for a in s["arguments"]} == {"pro", "contra"}
    assert all(a["cited"] for a in s["arguments"] if a["citation_valid"])
