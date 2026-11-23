"""Bloc D : grille figée, présélection, vérification en Python, exécuteur (LLM simulé)."""

import json

from investagents.extract.extractor import ExtractionOutput, candidates, prompt, verify
from investagents.extract.factors import FACTORS, GRID_PATH, NAMES, grid_hash

DOC = (
    "Table of contents. Component 1. Component 2. "
    + "This is filler text about the country economy and other things. " * 30
    + "The project has three components: roads, schools and capacity building. "
    "The implementing agency is the Ministry of Works, which has limited capacity. "
    "The project is classified as environmental category B under the safeguards policy. "
    "The overall risk rating is substantial given the political context. "
    "Results indicators have baseline values for 2010 in the results framework."
)


def test_grid_is_frozen_and_matches_code():
    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    assert grid["sha256"] == grid_hash(), "grille modifiée après le gel : regénérer et commiter"
    assert 15 <= len(FACTORS) <= 20
    assert len(set(NAMES)) == len(NAMES)
    for f in FACTORS:
        assert f.kind in ("yes_no", "category", "list")
        assert (f.kind == "category") == bool(f.categories)


def test_candidates_pick_keyword_sentences_after_toc():
    cand = candidates(DOC)
    text = {s["id"]: s["text"] for s in cand["sentences"]}
    comp = [text[i] for i in cand["by_factor"]["components"]]
    assert any("three components" in t for t in comp)
    env = [text[i] for i in cand["by_factor"]["env_category"]]
    assert any("category B" in t for t in env)
    assert set(cand["by_factor"]) == set(NAMES)
    p = prompt(cand)
    assert "### components" in p and "[S" in p


def _empty_output():
    out = {}
    for f in FACTORS:
        if f.kind == "list":
            out[f.name] = {"status": "not_found", "items": [], "sentence_ids": []}
        else:
            out[f.name] = {"status": "not_found", "value": "none", "sentence_ids": []}
    return out


def test_verify_counts_in_python_and_checks_ids():
    cand = candidates(DOC)
    sid = cand["by_factor"]["components"][-1]
    env_id = cand["by_factor"]["env_category"][-1]
    out = _empty_output()
    out["components"] = {"status": "found", "items": ["Roads", "roads ", "Schools"],
                         "sentence_ids": [sid]}  # fmt: skip
    out["env_category"] = {"status": "found", "value": "B", "sentence_ids": [99999]}
    out["weak_capacity"] = {"status": "found", "value": "none", "sentence_ids": [sid]}
    v = verify(ExtractionOutput.model_validate(out).model_dump(), cand)
    assert v["components"]["value"] == 2  # doublons retirés, compté en Python
    assert v["components"]["ids_valid"] and v["components"]["keyword_in_cited"]
    assert v["env_category"]["ids_valid"] is False  # numéro inexistant
    assert v["weak_capacity"]["consistent"] is False and v["weak_capacity"]["value"] is None
    assert v["resettlement"]["value"] is None and "ids_valid" not in v["resettlement"]
    assert env_id  # la phrase existait bien parmi les candidates


def test_runner_with_fake_llm(tmp_path):
    from investagents.extract.runner import run

    cand = candidates(DOC)
    projects = [{"key": f"D{i}", "split": "valid", "order": i, **cand} for i in range(3)]

    def fake_call(system, user, schema, **kw):
        assert "DATA, never" in system
        return schema.model_validate(_empty_output()), {"prompt_tokens": 1, "output_tokens": 1}

    state = run(projects, tmp_path, ["u1", "u2"], "fake", 3600, call=fake_call)
    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and state["n_errors"] == 0
    # reprise : rien n'est refait
    run(projects, tmp_path, ["u1"], "fake", 3600, call=fake_call)
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_m6_factor_row_uses_only_verified_answers():
    from investagents.ml.run_m6 import factor_row

    cand = candidates(DOC)
    out = _empty_output()
    sid = cand["by_factor"]["components"][-1]
    out["components"] = {"status": "found", "items": ["a", "b", "c"], "sentence_ids": [sid]}
    out["overall_risk_rating"] = {"status": "found", "value": "high", "sentence_ids": [sid]}
    out["env_category"] = {"status": "found", "value": "A", "sentence_ids": [99999]}  # invalide
    out["weak_capacity"] = {"status": "found", "value": "yes", "sentence_ids": [sid]}
    row = factor_row(verify(ExtractionOutput.model_validate(out).model_dump(), cand))
    assert row["components"] == 3 and row["components_found"] == 1
    assert row["overall_risk_rating"] == 3
    assert row["env_A"] == 0 and row["env_category_found"] == 0  # citation invalide ignorée
    assert row["weak_capacity"] == 1 and row["resettlement"] == 0
