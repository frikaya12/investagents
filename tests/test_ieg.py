"""Tests du module IEG : pagination, nettoyage, cible binaire, doublons."""

import pandas as pd
import pytest

from investagents.data import ieg


def row(pid, outcome, eval_fy="2010", approval_fy=2000, **kw):
    """Une ligne au format réel de l'API (types observés : evaluation_fy en texte)."""
    return {
        "project_id": pid,
        "outcome": outcome,
        "evaluation_fy": eval_fy,
        "approval_fy": approval_fy,
        "final_closing_fy": 2008.0,
        **kw,
    }


def test_fetch_paginates_until_short_page(monkeypatch):
    monkeypatch.setattr(ieg, "PAGE_SIZE", 2)
    pages = [
        {"count": 5, "data": [row("P000001", "Satisfactory"), row("P000002", "Satisfactory")]},
        {"count": 5, "data": [row("P000003", "Satisfactory"), row("P000004", "Satisfactory")]},
        {"count": 5, "data": [row("P000005", "Satisfactory")]},
    ]
    skips = []

    def fake_get(url, params):
        skips.append(params["skip"])
        return pages[len(skips) - 1]

    df = ieg.fetch_ieg(get=fake_get)
    assert len(df) == 5
    assert skips == [0, 2, 4]


def test_fetch_detects_incomplete_download(monkeypatch):
    monkeypatch.setattr(ieg, "PAGE_SIZE", 2)

    def fake_get(url, params):
        return {"count": 10, "data": [row("P000001", "Satisfactory")]}

    with pytest.raises(ValueError, match="incomplet"):
        ieg.fetch_ieg(get=fake_get)


def test_clean_binary_target_and_rank():
    raw = pd.DataFrame(
        [
            row("P000001", "Highly Satisfactory"),
            row("P000002", "Moderately Satisfactory"),
            row("P000003", "Moderately Unsatisfactory"),
            row("P000004", "Highly Unsatisfactory"),
            row("P000005", "Not Rated"),
            row("P000006", " Satisfactory "),  # espaces parasites
        ]
    )
    df = ieg.clean_ieg(raw).set_index("project_id")
    assert df.loc["P000001", "satisfactory"] == 1
    assert df.loc["P000002", "satisfactory"] == 1
    assert df.loc["P000003", "satisfactory"] == 0
    assert df.loc["P000004", "satisfactory"] == 0
    assert pd.isna(df.loc["P000005", "satisfactory"])
    assert df.loc["P000006", "outcome_rank"] == 5
    assert df.loc["P000004", "outcome_rank"] == 1
    assert df["evaluation_fy"].dtype == "Int64"


def test_clean_rejects_unknown_label():
    raw = pd.DataFrame([row("P000001", "Fairly Good")])
    with pytest.raises(ValueError, match="inconnus"):
        ieg.clean_ieg(raw)


def test_clean_keeps_latest_evaluation_for_duplicates():
    # Cas réel observé : P122764 apparaît deux fois (évaluations 2023 et 2026).
    raw = pd.DataFrame(
        [
            row("P122764", "Satisfactory", eval_fy="2026"),
            row("P122764", "Unsatisfactory", eval_fy="2023"),
        ]
    )
    df = ieg.clean_ieg(raw)
    assert len(df) == 1
    assert df.loc[0, "evaluation_fy"] == 2026
    assert df.loc[0, "outcome"] == "Satisfactory"


def test_summarize_counts():
    raw = pd.DataFrame(
        [row("P000001", "Satisfactory"), row("P000002", "Unsatisfactory"), row("P3", "Not Rated")]
    )
    s = ieg.summarize(ieg.clean_ieg(raw))
    assert s["n_projects"] == 3
    assert s["n_rated"] == 2
    assert s["outcome_counts"]["Satisfactory"] == 1
    assert s["outcome_counts"]["Not Rated"] == 1


@pytest.mark.network
def test_real_api_first_page_format():
    from investagents.data.http import get_json

    page = get_json(
        ieg.API_URL,
        {"datasetId": ieg.DATASET_ID, "resourceId": ieg.RESOURCE_ID, "top": 5, "type": "json"},
    )
    assert page["count"] > 10_000
    first = page["data"][0]
    for col in ["project_id", "outcome", "approval_fy", "evaluation_fy", "evaluation_type"]:
        assert col in first
