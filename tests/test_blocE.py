"""Bloc E : la liste des noms de pays est figée, reproductible et bien étiquetée (sans résultat)."""

import json

import pytest

from investagents.ml.country_terms import MLEDOZE, OUT, build


def test_terms_frozen_and_labels():
    terms = json.loads(OUT.read_text(encoding="utf-8"))
    for w in ("indonesia", "lanka", "guinea", "zaire", "chad"):
        assert w in terms["country_tokens"]
    for w in ("indonesian", "chinese", "french", "cameroonian"):
        assert w in terms["adjective_tokens"]
    for w in ("republic", "city", "south", "africa"):
        assert w not in terms["country_tokens"] and w not in terms["adjective_tokens"]
    assert "Africa" in terms["column_values_excluded_as_regions"]
    assert "Republic of Korea" in terms["column_values_kept"]


def test_terms_rebuild_identically():
    if not MLEDOZE.exists():
        pytest.skip("liste standard locale absente (data/raw non versionné)")
    terms = json.loads(OUT.read_text(encoding="utf-8"))
    assert build()["sha256"] == terms["sha256"]
