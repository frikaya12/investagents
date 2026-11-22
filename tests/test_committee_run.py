"""Tests de l'exécuteur Kaggle du bloc C (graphe et LLM simulés, sans réseau)."""

import json

from investagents.committee_run import runner as rn


class FakeGraph:
    def __init__(self, fail_keys=()):
        self.calls = []
        self.fail_keys = set(fail_keys)

    def invoke(self, state):
        self.calls.append((state["masked_id"], state["ollama_url"]))
        if state["masked_id"] in self.fail_keys:
            raise RuntimeError("serveur tombé")
        return {
            "judge": {"scores": []},
            "debate": {"arguments": []},
            "citations": [{"valid": True}, {"valid": False}],
            "errors": [],
        }


def projects(n):
    return [
        {"key": f"C{i:04d}", "split": "valid" if i % 2 else "test", "order": i,
         "sentences": [{"id": 1, "theme": "objectives", "text": "The objective is x."}]}
        for i in range(n)
    ]  # fmt: skip


def make(tmp_path, n=6, graph=None, deadline=3600, urls=("u1", "u2")):
    g = graph or FakeGraph()
    return rn.Runner(projects(n), tmp_path, list(urls), "m", 1, deadline, graph=g)


def test_batch_saves_every_project_and_uses_all_servers(tmp_path):
    g = FakeGraph()
    r = make(tmp_path, graph=g)
    r.run_batch(r.projects)
    lines = rn.load_jsonl(tmp_path / "results.jsonl")
    assert sorted(x["key"] for x in lines) == [f"C{i:04d}" for i in range(6)]
    assert lines[0]["citation_valid_rate"] == 0.5
    assert {u for _, u in g.calls} == {"u1", "u2"}
    prog = json.loads((tmp_path / "progress.json").read_text())
    assert prog["n_done"] == 6 and prog["n_with_errors"] == 0


def test_resume_skips_done_and_retries_failed(tmp_path):
    g = FakeGraph(fail_keys={"C0002"})
    r = make(tmp_path, graph=g)
    r.run_batch(r.projects)
    g2 = FakeGraph()
    r2 = make(tmp_path, graph=g2)
    r2.run_batch(r2.projects)
    assert [k for k, _ in g2.calls] == ["C0002"]  # seul l'échec est retenté


def test_stops_before_deadline(tmp_path):
    r = make(tmp_path, deadline=1)  # 1 s < 2 × 600 s (durée supposée avant toute mesure)
    r.run_batch(r.projects)
    assert not (tmp_path / "results.jsonl").exists() and r.stopped_for_time


def test_load_jsonl_ignores_truncated_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"key": "a"}\n{"key": "b', encoding="utf-8")
    assert rn.load_jsonl(p) == [{"key": "a"}]


def test_recognition_writes_one_line_per_project(tmp_path, monkeypatch):
    from investagents.schemas.committee import RecognitionOutput

    def fake_call(system, user, schema, **kw):
        assert "C000" not in user  # la clé n'est pas montrée au modèle
        data = {"recognized": False, "project_name_guess": "", "country_guess": "",
                "approval_year_guess": None, "outcome_guess": "unknown"}  # fmt: skip
        return RecognitionOutput.model_validate(data), {}

    monkeypatch.setattr(rn, "call_json", fake_call)
    r = make(tmp_path)
    r.recognition(n=4)
    assert len(rn.load_jsonl(tmp_path / "recognition.jsonl")) == 4


def test_m7_critique_features_counts():
    from investagents.ml.run_m7 import critique_features

    initial = [
        {"criterion": "risk", "stance": "pro", "claim": "a", "sentence_ids": [1]},
        {"criterion": "risk", "stance": "contra", "claim": "b", "sentence_ids": [2]},
    ]
    r = {
        "initial_debate": {"arguments": initial},
        "rounds_history": [{"critiques": [
            {"argument_index": 0, "verdict": "strong", "comment": ""},
            {"argument_index": 1, "verdict": "unsupported", "comment": ""},
        ]}],
        "debate": {"arguments": [initial[0], {**initial[1], "claim": "b réécrit"}]},
    }  # fmt: skip
    f = critique_features(r)
    assert f["strong_pro_risk"] == 1 and f["strong_contra_risk"] == 0
    assert f["share_unsupported"] == 0.5
    assert f["share_rewritten"] == 0.5
    assert f["contra_text"] == "b réécrit"  # texte Contre final, après raffinement
