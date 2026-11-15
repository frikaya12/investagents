"""Tests du corpus : choix du document principal, cache gzip."""

import gzip

import pandas as pd

from investagents.data import corpus as co


def test_select_main_document_latest_tier1_with_url():
    kept = pd.DataFrame(
        {
            "project_id": ["P1", "P1", "P1", "P2", "P3"],
            "doc_id": ["a", "b", "c", "d", "e"],
            "tier": [1, 1, 2, 1, 1],
            "docdt": pd.to_datetime(
                ["2010-01-01", "2010-03-01", "2010-04-01", "2001-01-01", "2002-01-01"]
            ),
            "txturl": ["u", "u", "u", "u", None],
        }
    )
    out = co.select_main_documents(kept).set_index("project_id")
    assert out.loc["P1", "doc_id"] == "b"  # le plus récent des tier 1 (pas le tier 2)
    assert out.loc["P2", "doc_id"] == "d"
    assert "P3" not in out.index  # pas de txturl


def test_download_retries_after_403(tmp_path, monkeypatch):
    class Resp:
        def __init__(self, status, text=""):
            self.status_code, self.text, self.encoding = status, text, None

    answers = [Resp(403), Resp(403), Resp(200, "x" * 20)]
    monkeypatch.setattr(co.requests, "get", lambda *a, **k: answers.pop(0))
    waits = []
    n = co.download_gz("D9", "http://u", text_dir=tmp_path, sleep=waits.append)
    assert n == 20
    assert waits == [15, 30]
    assert co.read_text("D9", text_dir=tmp_path) == "x" * 20


def test_download_uses_gzip_cache(tmp_path):
    with gzip.open(tmp_path / "D1.txt.gz", "wt", encoding="utf-8") as f:
        f.write("hello world")
    assert co.download_gz("D1", "http://ignored", text_dir=tmp_path) == 11
    assert co.read_text("D1", text_dir=tmp_path) == "hello world"
