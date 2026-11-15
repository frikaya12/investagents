"""Tests du montant à l'approbation : prêts ultérieurs exclus, formats réels."""

import pandas as pd

from investagents.data import amounts as am


def test_standardize_real_formats():
    raw = pd.DataFrame(
        {
            "project_id": ["P007335", None],
            "board_approval_date": ["11-May-1961", "01-Jan-2000"],
            "original_principal_amount_us_": ["9000000", "1"],
        }
    )
    out = am.standardize(raw, "IDA", "original_principal_amount_us_")
    assert len(out) == 1  # ligne sans project_id retirée
    assert out.iloc[0]["approval_date"] == pd.Timestamp("1961-05-11")
    assert out.iloc[0]["original_amount_usd"] == 9_000_000


def test_amount_excludes_later_loans():
    loans = pd.DataFrame(
        {
            "project_id": ["P1", "P1", "P1", "P2"],
            "source": ["IDA", "IBRD", "IDA", "IBRD"],
            "approval_date": pd.to_datetime(
                ["2010-05-01", "2010-05-01", "2014-03-01", "2001-01-01"]
            ),
            "original_amount_usd": [50.0, 30.0, 100.0, 7.0],
        }
    )
    out = am.amount_at_approval(loans).set_index("project_id")
    assert out.loc["P1", "amount_at_approval_usd"] == 80.0  # IDA + IBRD du même jour
    assert out.loc["P1", "n_later_loans"] == 1  # le prêt de 2014 est exclu
    assert out.loc["P2", "amount_at_approval_usd"] == 7.0


def test_fetch_detects_incomplete(monkeypatch):
    monkeypatch.setattr(am, "PAGE_SIZE", 2)

    def fake_get(url, params):
        return {"count": 5, "data": [{"project_id": "P1"}]}

    import pytest

    with pytest.raises(ValueError):
        am.fetch_statement("DS", "RS", get=fake_get)
