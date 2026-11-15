"""Tests de la Projects API : pagination et parsing des formats réels observés."""

import pandas as pd
import pytest

from investagents.data import projects_api as pa

# Ligne au format réel (copiée de la réponse pour P122764, champs utiles seulement).
REAL_ROW = {
    "id": "P122764",
    "boardapprovaldate": "2012-05-24T00:00:00Z",
    "approvalfy": "2012",
    "closingdate": "12/31/2024 12:00:00 AM",
    "totalamt": "50,000,000",
    "lendinginstr": "Investment Project Financing",
    "regionname": "Eastern and Southern Africa",
    "countryname": ["Federal Democratic Republic of Ethiopia"],
    "status": "Active",
}


def test_parse_real_format():
    df = pa.parse_projects([REAL_ROW])
    r = df.iloc[0]
    assert r["approval_date"] == pd.Timestamp("2012-05-24")
    assert r["closing_date"] == pd.Timestamp("2024-12-31")
    assert r["total_amount_usd"] == 50_000_000.0
    assert r["approval_fy_api"] == 2012
    assert r["country"] == "Federal Democratic Republic of Ethiopia"


def test_parse_missing_and_bad_values():
    row = {"id": "P000009", "boardapprovaldate": "", "totalamt": "n/a", "countryname": "Chad"}
    r = pa.parse_projects([row]).iloc[0]
    assert pd.isna(r["approval_date"])
    assert r["total_amount_usd"] is None or pd.isna(r["total_amount_usd"])
    assert r["country"] == "Chad"


def test_parse_drops_duplicates():
    df = pa.parse_projects([REAL_ROW, REAL_ROW])
    assert len(df) == 1


def test_fetch_paginates(monkeypatch):
    monkeypatch.setattr(pa, "PAGE_SIZE", 2)
    offsets = []

    def fake_get(url, params):
        offsets.append(params["os"])
        n = {0: 2, 2: 1}[params["os"]]
        ids = [f"P00000{params['os'] + i}" for i in range(n)]
        return {"total": "3", "projects": {i: {"id": i} for i in ids}}

    rows = pa.fetch_all_projects(get=fake_get)
    assert len(rows) == 3
    assert offsets == [0, 2]


@pytest.mark.network
def test_real_api_one_project():
    from investagents.data.http import get_json

    page = get_json(pa.API_URL, {"format": "json", "id": "P122764"})
    p = page["projects"]["P122764"]
    assert p["boardapprovaldate"].startswith("2012-05-24")
