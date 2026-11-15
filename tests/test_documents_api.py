"""Tests de l'API Documents : pagination, parsing, multi-projets, échantillon, texte."""

import pandas as pd
import pytest

from investagents.data import documents_api as da

REAL_DOC = {  # format réel (PAD de P122764), champs utiles seulement
    "docty": "Project Appraisal Document",
    "docdt": "2012-04-26T04:00:00Z",
    "projectid": "P122764",
    "lang": "English",
    "display_title": "Ethiopia - Women Entrepreneurship Development Project",
    "txturl": "http://documents.worldbank.org/curated/en/330121468024613142/text/x.txt",
}


def test_fetch_paginates_and_skips_facets(monkeypatch):
    monkeypatch.setattr(da, "PAGE_SIZE", 2)

    def fake_get(url, params):
        pages = {
            0: {"total": "3", "documents": {"D1": REAL_DOC, "D2": REAL_DOC, "facets": {}}},
            2: {"total": "3", "documents": {"D3": REAL_DOC, "facets": {}}},
        }
        assert params["projectid"] == "P122764"
        return pages[params["os"]]

    docs = da.fetch_project_documents("P122764", get=fake_get)
    assert [d["doc_id"] for d in docs] == ["D1", "D2", "D3"]


def test_parse_dates_and_fields():
    df = da.parse_documents([{"doc_id": "D1", **REAL_DOC}])
    r = df.iloc[0]
    assert r["docdt"] == pd.Timestamp("2012-04-26")  # jour, sans fuseau
    assert r["project_id"] == "P122764"
    assert r["docty"] == "Project Appraisal Document"


def test_parse_splits_multi_project_documents():
    doc = {"doc_id": "D9", **REAL_DOC, "projectid": "P514478,P518183"}
    df = da.parse_documents([doc])
    assert sorted(df["project_id"]) == ["P514478", "P518183"]


def test_parse_handles_missing_values():
    df = da.parse_documents([{"doc_id": "D1", "docty": "Report", "txturl": ""}])
    r = df.iloc[0]
    assert pd.isna(r["docdt"])
    assert r["txturl"] is None
    assert r["project_id"] is None


def test_sample_is_reproducible_and_rated_only():
    ieg = pd.DataFrame(
        {
            "project_id": [f"P{i:06d}" for i in range(200)],
            "satisfactory": [pd.NA if i % 4 == 0 else i % 2 for i in range(200)],
        }
    )
    a = da.sample_rated_projects(ieg, n=20, seed=1)
    b = da.sample_rated_projects(ieg, n=20, seed=1)
    assert a["project_id"].tolist() == b["project_id"].tolist()
    assert a["satisfactory"].notna().all()


def test_download_text_uses_cache(tmp_path):
    (tmp_path / "D1.txt").write_text("abc", encoding="utf-8")
    assert da.download_text("http://ignored", "D1", text_dir=tmp_path) == 3


@pytest.mark.parametrize("url", [None, float("nan"), ""])
def test_download_text_without_url(tmp_path, url):
    assert da.download_text(url, "D3", text_dir=tmp_path) == 0


def test_download_text_returns_zero_on_error(tmp_path, monkeypatch):
    class Resp:
        status_code = 404

    monkeypatch.setattr(da.requests, "get", lambda *a, **k: Resp())
    assert da.download_text("http://x", "D2", text_dir=tmp_path) == 0
    assert not (tmp_path / "D2.txt").exists()


@pytest.mark.network
def test_real_api_filters():
    from investagents.data.http import get_json

    base = {"format": "json", "rows": 1, "fl": "docty"}
    pad = get_json(da.API_URL, {**base, "docty_exact": "Project Appraisal Document"})
    assert int(pad["total"]) > 1000
    none = get_json(da.API_URL, {**base, "docty_exact": "Nonexistent Type XYZ"})
    assert int(none["total"]) == 0
    one = get_json(
        da.API_URL,
        {**base, "docty_exact": "Project Appraisal Document", "projectid": "P122764"},
    )
    assert int(one["total"]) == 1
