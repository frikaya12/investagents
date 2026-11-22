"""Test de fumée de la démo : l'application s'exécute sans exception (sans navigateur)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "investagents" / "app" / "streamlit_app.py"


def test_portfolio_page_runs():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception


def test_projects_page_runs():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio[0].set_value("Projets analysés").run()
    assert not at.exception


def test_multicriteria_page_runs():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Sélection multicritère (comité)").run()
    assert not at.exception
