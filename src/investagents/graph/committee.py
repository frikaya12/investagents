"""Graphe LangGraph du comité compact.

    extraire (Python) -> débattre (LLM, 1 appel) -> juger (LLM, 1 appel) -> vérifier (Python)

- extraire : sections clés ≤ 8 000 caractères, identifiants masqués, découpées en phrases
  numérotées [S1], [S2]... (agents/sections.py) ;
- débattre : Pour et Contre pour chacun des 5 critères ; chaque argument cite des NUMÉROS de
  phrases (le modèle ne recopie plus de texte, il ne peut donc plus paraphraser une citation) ;
- juger : 5 notes de 0 à 10 à partir des seuls arguments et des phrases citées valides ;
- vérifier : chaque numéro doit exister ; la phrase correspondante doit être une sous-chaîne
  exacte du document complet (masqué). Aucun chiffre n'est produit par le LLM hors des notes.
Une sortie invalide deux fois de suite est consignée dans `errors` (signalée à l'humain), le
graphe s'arrête proprement au lieu de planter.
Inspiré du graphe de DIALECTIC (`src/agent/pipeline/graph.py`, MIT) en version réduite : pas de
décomposition en questions, pas de recherche web (interdite ici), 1 tour, pas de raffinement.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from investagents.agents.citations import quote_found
from investagents.agents.llm import DEFAULT_MODEL, LLMOutputError, call_json
from investagents.agents.sections import (
    extract_key_sections,
    mask_identifiers,
    normalize,
    number_sentences,
    sentences_to_prompt,
)
from investagents.debate.prompts import (
    CRITERIA_TEXT,
    DEBATE_SYSTEM,
    DEBATE_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
)
from investagents.schemas.committee import DebateOutput, JudgeOutput


class CommitteeState(TypedDict, total=False):
    masked_id: str
    raw_text: str
    project_name: str
    model: str
    masked_text: str
    sections: dict[str, str]
    sentences: list[dict[str, Any]]
    agent_text: str
    debate: dict[str, Any] | None
    judge: dict[str, Any] | None
    citations: list[dict[str, Any]]
    llm_stats: dict[str, Any]
    errors: list[str]


def extract(state: CommitteeState) -> dict:
    text = mask_identifiers(normalize(state["raw_text"]), name=state.get("project_name"))
    sections = extract_key_sections(text)
    sentences = number_sentences(sections)
    return {
        "masked_text": text,
        "sections": sections,
        "sentences": sentences,
        "agent_text": sentences_to_prompt(sentences),
        "errors": [],
    }


def debate(state: CommitteeState) -> dict:
    try:
        out, stats = call_json(
            DEBATE_SYSTEM,
            DEBATE_USER.format(criteria=CRITERIA_TEXT, document=state["agent_text"]),
            DebateOutput,
            model=state.get("model", DEFAULT_MODEL),
            temperature=0.5,
            num_predict=1800,
        )
    except LLMOutputError as exc:
        return {"debate": None, "errors": [*state.get("errors", []), f"débat : {exc}"]}
    return {"debate": out.model_dump(), "llm_stats": {"debate": stats}}


def _cited_text(ids: list[int], sentences: list[dict]) -> str:
    by_id = {s["id"]: s["text"] for s in sentences}
    return " ".join(f"[S{i}] {by_id[i]}" for i in ids if i in by_id)


def judge(state: CommitteeState) -> dict:
    if not state.get("debate"):
        return {"judge": None}
    sentences = state.get("sentences", [])
    args = "\n".join(
        f"- {a['criterion']} | {a['stance']} | {a['claim']} | "
        f"{_cited_text(a['sentence_ids'], sentences) or '(no valid citation)'}"
        for a in state["debate"]["arguments"]
    )
    try:
        out, stats = call_json(
            JUDGE_SYSTEM,
            JUDGE_USER.format(criteria=CRITERIA_TEXT, arguments=args),
            JudgeOutput,
            model=state.get("model", DEFAULT_MODEL),
            temperature=0.0,
            num_predict=900,
        )
    except LLMOutputError as exc:
        return {"judge": None, "errors": [*state.get("errors", []), f"juge : {exc}"]}
    return {"judge": out.model_dump(), "llm_stats": {**state.get("llm_stats", {}), "judge": stats}}


def verify(state: CommitteeState) -> dict:
    """Une citation (argument) est valide si TOUS ses numéros existent et que chaque phrase
    citée se retrouve mot pour mot dans le document complet."""
    if not state.get("debate"):
        return {"citations": []}
    by_id = {s["id"]: s["text"] for s in state.get("sentences", [])}
    full = state.get("masked_text", normalize(state["raw_text"]))
    checks = []
    for a in state["debate"]["arguments"]:
        ids = a["sentence_ids"]
        known = [i for i in ids if i in by_id]
        in_doc = all(quote_found(by_id[i], full) for i in known) if known else False
        checks.append(
            {
                "criterion": a["criterion"],
                "stance": a["stance"],
                "sentence_ids": ids,
                "unknown_ids": [i for i in ids if i not in by_id],
                "cited_text": [by_id[i] for i in known],
                "valid": len(known) == len(ids) and in_doc,
            }
        )
    return {"citations": checks}


def build_committee():
    g = StateGraph(CommitteeState)
    g.add_node("extract", extract)
    g.add_node("debate", debate)
    g.add_node("judge", judge)
    g.add_node("verify", verify)
    g.add_edge(START, "extract")
    g.add_edge("extract", "debate")
    g.add_edge("debate", "judge")
    g.add_edge("judge", "verify")
    g.add_edge("verify", END)
    return g.compile()
