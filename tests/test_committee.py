"""Tests du comité sans LLM réel : sections, masquage, citations, schémas, graphe (LLM simulé)."""

import json

import pytest
from pydantic import ValidationError

from investagents.agents import citations as ct
from investagents.agents import llm
from investagents.agents import sections as sc
from investagents.graph import committee as cm
from investagents.schemas.committee import CRITERIA, DebateOutput, JudgeOutput

DOC = (
    "Table of contents: development objective, risks. "
    + "x " * 400
    + "The project development objective is to improve access to water in rural areas. "
    + "Key risks include weak capacity of the implementing agency and currency depreciation. "
    + "The economic analysis shows an ERR of 14 percent. Project P123456 is called Rural Water."
)


def full_debate():
    return {
        "arguments": [
            {
                "criterion": c,
                "stance": s,
                "claim": f"{s} argument about {c}.",
                "quote": "improve access to water in rural areas",
            }
            for c in CRITERIA
            for s in ("pro", "contra")
        ]
    }


def full_judge():
    return {
        "scores": [
            {"criterion": c, "score": 6, "justification": "Balanced view."} for c in CRITERIA
        ]
    }


def test_sections_skip_table_of_contents_and_respect_length():
    s = sc.extract_key_sections(DOC, max_chars=600)
    assert sum(len(v) for v in s.values()) <= 600
    assert s["objectives"].startswith("development objective is to improve")
    assert "currency" in s["risks"]


def test_sections_are_exact_substrings():
    t = sc.normalize(DOC)
    for v in sc.extract_key_sections(DOC, 600).values():
        assert v in t


def test_mask_identifiers():
    out = sc.mask_identifiers(DOC, name="Rural Water")
    assert "P123456" not in out and "[PROJECT_ID]" in out
    assert "Rural Water" not in out


def test_quote_found_normalizes_spacing_case_and_quotes():
    assert ct.quote_found("Key  RISKS include weak capacity", DOC)
    assert ct.quote_found("“Key risks include weak capacity”", DOC)
    assert not ct.quote_found("Key risks include strong capacity", DOC)
    assert not ct.quote_found("Key", DOC)  # trop court
    assert ct.citation_rate(["Key risks include weak", "invented sentence here"], DOC) == 0.5


def test_debate_schema_requires_pro_and_contra_for_each_criterion():
    DebateOutput.model_validate(full_debate())
    bad = full_debate()
    bad["arguments"] = [a for a in bad["arguments"] if a["criterion"] != "risk"]
    with pytest.raises(ValidationError):
        DebateOutput.model_validate(bad)


def test_judge_schema_bounds():
    JudgeOutput.model_validate(full_judge())
    bad = full_judge()
    bad["scores"][0]["score"] = 11
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate(bad)


class FakeResp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self.content}, "prompt_eval_count": 10, "eval_count": 5}


def test_call_json_retries_once_then_fails():
    calls = []

    def post(url, json, timeout):
        calls.append(json["options"]["seed"])
        return FakeResp("{not json")

    with pytest.raises(llm.LLMOutputError):
        llm.call_json("s", "u", JudgeOutput, post=post)
    assert len(calls) == 2


def test_call_json_second_attempt_succeeds():
    answers = [FakeResp("{}"), FakeResp(json.dumps(full_judge()))]
    out, stats = llm.call_json("s", "u", JudgeOutput, post=lambda *a, **k: answers.pop(0))
    assert stats["attempts"] == 2 and len(out.scores) == 5


def test_graph_end_to_end_with_fake_llm(monkeypatch):
    def fake_call(system, user, schema, **kw):
        assert "P123456" not in user  # identifiant masqué côté agents
        data = full_debate() if schema is DebateOutput else full_judge()
        return schema.model_validate(data), {"attempts": 1}

    monkeypatch.setattr(cm, "call_json", fake_call)
    state = cm.build_committee().invoke({"raw_text": DOC, "project_name": "Rural Water"})
    assert len(state["judge"]["scores"]) == 5
    assert all(c["found_in_document"] for c in state["citations"])
    assert state["errors"] == []


def test_graph_flags_invalid_output(monkeypatch):
    def failing(*a, **k):
        raise llm.LLMOutputError("boom")

    monkeypatch.setattr(cm, "call_json", failing)
    state = cm.build_committee().invoke({"raw_text": DOC})
    assert state["judge"] is None and state["errors"]
