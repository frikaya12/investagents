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


def fr_ci(d: dict, signed: bool = False) -> str:
    """Mise en forme indépendante : « 0,297 [0,245 ; 0,370] »."""

    def fr(x):
        return ("−" if x < 0 else "") + f"{abs(x):.3f}".replace(".", ",")

    v = fr(d["value"])
    if signed and d["value"] >= 0:
        v = "+" + v
    return f"{v} [{fr(d['ci_low'])} ; {fr(d['ci_high'])}]"


def test_bloc_e_report_numbers_come_from_json():
    """Rapport du bloc E à jour ; chiffres relus ici dans le JSON, sans passer par metrics.py."""
    from investagents.report.build import BLOC_E_OUT, render_bloc_e
    from investagents.report.metrics import ROOT, load

    res = ROOT / "experiments/blocE/results.json"
    if not res.exists():
        pytest.skip("bloc E non calculé")
    e = json.loads(res.read_text(encoding="utf-8"))
    assert BLOC_E_OUT.read_text(encoding="utf-8") == render_bloc_e(load()), "relancer le build"
    d = e["diff_no_country_minus_original"]
    expected = [fr_ci(e["original"]["auc_pr"]), fr_ci(e["no_country"]["auc_pr"]),
                fr_ci(e["original"]["auc_roc"]), fr_ci(e["no_country"]["auc_roc"]),
                fr_ci(d["auc_pr"], signed=True), fr_ci(d["auc_roc"], signed=True)]  # fmt: skip
    final = (ROOT / "reports/rapport_final.md").read_text(encoding="utf-8")
    resume = (ROOT / "docs/resume_prof.md").read_text(encoding="utf-8")
    for s in expected:
        assert s in BLOC_E_OUT.read_text(encoding="utf-8") and s in final
    assert fr_ci(e["no_country"]["auc_pr"]) in resume and fr_ci(d["auc_pr"], signed=True) in resume
    assert e["verdict"] in final and e["verdict"] in BLOC_E_OUT.read_text(encoding="utf-8")
    # les mots cités dans la nuance sont bien parmi les mots les plus lourds
    top = {w["word"] for side in ("toward_failure", "toward_success") for w in e["words"][side]}
    assert {"nairobi", "africa", "sofide", "rs"} <= top
