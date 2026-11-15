"""Tests des indicateurs pays-année : décalage ex ante, ancienneté maximale, pays du projet."""

import pandas as pd
import pytest

from investagents.data import country_indicators as ci

LONG = pd.DataFrame(
    {
        "iso3": ["ETH"] * 5 + ["KEN"],
        "year": [2005, 2008, 2009, 2010, 2011, 2000],
        "indicator": ["cpia"] * 6,
        "value": [3.0, 3.1, 3.2, 3.3, 3.4, 9.9],
    }
)


def test_as_of_join_uses_value_two_years_before_approval():
    projects = pd.DataFrame({"project_id": ["P1"], "iso3": ["ETH"], "approval_fy": [2011]})
    out = ci.as_of_join(projects, LONG)
    # exercice 2011 -> année cible 2009 : la valeur 2010 et 2011 ne doivent PAS être utilisées
    assert out.loc[0, "cpia"] == 3.2


def test_as_of_join_carries_last_value_forward():
    projects = pd.DataFrame({"project_id": ["P1"], "iso3": ["ETH"], "approval_fy": [2009]})
    # cible 2007 : pas de valeur 2006/2007 -> dernière connue = 2005
    assert ci.as_of_join(projects, LONG).loc[0, "cpia"] == 3.0


def test_as_of_join_rejects_stale_values():
    projects = pd.DataFrame({"project_id": ["P1"], "iso3": ["KEN"], "approval_fy": [2010]})
    # cible 2008, dernière valeur 2000 : 8 ans > 4 ans -> manquant
    assert pd.isna(ci.as_of_join(projects, LONG).loc[0, "cpia"])


def test_as_of_join_no_future_value():
    projects = pd.DataFrame({"project_id": ["P1"], "iso3": ["ETH"], "approval_fy": [2006]})
    # cible 2004 : la première valeur est 2005 (future) -> manquant
    assert pd.isna(ci.as_of_join(projects, LONG).loc[0, "cpia"])


def test_as_of_join_keeps_projects_without_country():
    projects = pd.DataFrame(
        {"project_id": ["P1", "P2"], "iso3": ["ETH", None], "approval_fy": [2011, 2011]}
    )
    out = ci.as_of_join(projects, LONG).set_index("project_id")
    assert len(out) == 2
    assert pd.isna(out.loc["P2", "cpia"])


def test_project_countries_most_frequent_code():
    s1 = pd.DataFrame({"project_id": ["P1", "P1", "P2"], "country_code": ["ET", "ET", "3A"]})
    s2 = pd.DataFrame({"project_id": ["P1"], "country_code": ["KE"]})
    out = ci.project_countries([s1, s2], {"ET": "ETH", "KE": "KEN"}).set_index("project_id")
    assert out.loc["P1", "iso3"] == "ETH"
    assert "P2" not in out.index  # code régional (« 3A ») écarté


def test_fetch_wdi_paginates_by_window(monkeypatch):
    monkeypatch.setattr(ci, "YEAR_WINDOWS", [(2000, 2004), (2005, 2009)])
    calls = []

    def fake_get(url, params):
        calls.append((params["date"], params["page"]))
        start = int(params["date"].split(":")[0])
        year = start + params["page"]
        row = {"country": {"id": "ET"}, "countryiso3code": "ETH", "date": str(year)}
        return [{"pages": 2}, [{**row, "value": 1.0}, {**row, "value": None}]]

    df, iso = ci.fetch_wdi("X", get=fake_get)
    assert calls == [("2000:2004", 1), ("2000:2004", 2), ("2005:2009", 1), ("2005:2009", 2)]
    assert df["year"].tolist() == [2001, 2002, 2006, 2007]
    assert iso == {"ET": "ETH"}


@pytest.mark.network
def test_real_wdi_indicator_exists():
    from investagents.data.http import get_json

    r = get_json(f"{ci.API}/indicator/IQ.CPA.IRAI.XQ", {"format": "json"})
    assert "IDA resource allocation index" in r[1][0]["name"]
