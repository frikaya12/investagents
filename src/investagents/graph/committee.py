"""Graphe LangGraph du comité compact (MVP).

    extraire (Python) -> débattre (LLM, 1 appel) -> juger (LLM, 1 appel) -> vérifier (Python)

- extraire : sections clés ≤ 8 000 caractères, identifiants masqués (agents/sections.py) ;
- débattre : Pour et Contre pour chacun des 5 critères, avec citation exacte ;
- juger : 5 notes de 0 à 10 à partir des seuls arguments ;
- vérifier : chaque citation est recherchée dans le texte donné aux agents ET dans le document
  complet (agents/citations.py). Aucun chiffre n'est produit par le LLM hors des notes du Juge.
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
    sections_to_prompt,
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
    sections: dict[str, str]
    agent_text: str
    debate: dict[str, Any] | None
    judge: dict[str, Any] | None
    citations: list[dict[str, Any]]
    llm_stats: dict[str, Any]
    errors: list[str]


def extract(state: CommitteeState) -> dict:
    text = mask_identifiers(normalize(state["raw_text"]), name=state.get("project_name"))
    sections = extract_key_sections(text)
    return {"sections": sections, "agent_text": sections_to_prompt(sections), "errors": []}


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


def judge(state: CommitteeState) -> dict:
    if not state.get("debate"):
        return {"judge": None}
    args = "\n".join(
        f'- {a["criterion"]} | {a["stance"]} | {a["claim"]} | "{a["quote"]}"'
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
    if not state.get("debate"):
        return {"citations": []}
    full = normalize(state["raw_text"])
    checks = [
        {
            "criterion": a["criterion"],
            "stance": a["stance"],
            "quote": a["quote"],
            "found_in_agent_text": quote_found(a["quote"], state["agent_text"]),
            "found_in_document": quote_found(a["quote"], full),
        }
        for a in state["debate"]["arguments"]
    ]
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
