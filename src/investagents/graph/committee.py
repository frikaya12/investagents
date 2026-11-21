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

Comité complet (bloc C, `build_full_committee`) :

    extraire -> débattre -> [critiquer -> raffiner] x tours -> juger (grille ancrée) -> vérifier

- critiquer : un critique donne un verdict (strong / weak / unsupported) sur chaque argument ;
- raffiner : les analystes réécrivent les 10 arguments en tenant compte des verdicts ;
- juger : notes 0–10 avec une grille ancrée (exemples de 2, 5 et 8 par critère).
`rounds` = nombre de tours critique + raffinement (0 = ablation « sans critique »).
Si les phrases numérotées sont déjà fournies (Kaggle), l'extraction est sautée ; sans texte
complet, la vérification ne contrôle que l'existence des numéros (`full_text_checked` = False) et
la vérification mot pour mot est refaite en local.
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
    ANCHORED_RUBRIC,
    CRITERIA_TEXT,
    CRITIQUE_SYSTEM,
    CRITIQUE_USER,
    DEBATE_SYSTEM,
    DEBATE_USER,
    JUDGE_ANCHORED_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
    REFINE_SYSTEM,
    REFINE_USER,
)
from investagents.schemas.committee import CritiqueOutput, DebateOutput, JudgeOutput


class CommitteeState(TypedDict, total=False):
    masked_id: str
    raw_text: str
    project_name: str
    model: str
    ollama_url: str
    num_ctx: int
    rounds: int
    rounds_history: list[dict[str, Any]]
    initial_debate: dict[str, Any] | None
    masked_text: str
    sections: dict[str, str]
    sentences: list[dict[str, Any]]
    agent_text: str
    debate: dict[str, Any] | None
    judge: dict[str, Any] | None
    citations: list[dict[str, Any]]
    llm_stats: dict[str, Any]
    errors: list[str]


def _llm(state: CommitteeState) -> dict:
    kw = {"model": state.get("model", DEFAULT_MODEL)}
    if state.get("ollama_url"):
        kw["url"] = state["ollama_url"]
    if state.get("num_ctx"):  # fenêtre de contexte (Ollama tronque le début si elle déborde)
        kw["num_ctx"] = state["num_ctx"]
    return kw


def extract(state: CommitteeState) -> dict:
    if state.get("sentences"):  # phrases déjà numérotées (jeu de données Kaggle)
        return {
            "agent_text": state.get("agent_text") or sentences_to_prompt(state["sentences"]),
            "errors": [],
        }
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
            **_llm(state),
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
            **_llm(state),
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
    full = state.get("masked_text") or (
        normalize(state["raw_text"]) if state.get("raw_text") else None
    )
    checks = []
    for a in state["debate"]["arguments"]:
        ids = a["sentence_ids"]
        known = [i for i in ids if i in by_id]
        if full is None:  # pas de texte complet ici : contrôle refait en local
            in_doc = bool(known)
        else:
            in_doc = all(quote_found(by_id[i], full) for i in known) if known else False
        checks.append(
            {
                "criterion": a["criterion"],
                "stance": a["stance"],
                "sentence_ids": ids,
                "unknown_ids": [i for i in ids if i not in by_id],
                "cited_text": [by_id[i] for i in known],
                "valid": len(known) == len(ids) and in_doc,
                "full_text_checked": full is not None,
            }
        )
    return {"citations": checks}


def _format_args(args: list[dict], sentences: list[dict], critiques=None) -> str:
    by_index = {c["argument_index"]: c for c in (critiques or [])}
    lines = []
    for i, a in enumerate(args):
        line = (
            f"{i} | {a['criterion']} | {a['stance']} | {a['claim']} | "
            f"{_cited_text(a['sentence_ids'], sentences) or '(no valid citation)'}"
        )
        if critiques is not None:
            c = by_index.get(i)
            line += f" | {c['verdict']} | {c['comment']}" if c else " | (no verdict) |"
        lines.append(line)
    return "\n".join(lines)


def critique_refine(state: CommitteeState) -> dict:
    """`rounds` tours de critique puis raffinement. En cas d'échec d'un appel, on garde les
    derniers arguments valides et on consigne l'erreur."""
    if not state.get("debate"):
        return {}
    sentences = state.get("sentences", [])
    current, history = state["debate"], []
    errors = list(state.get("errors", []))
    stats = dict(state.get("llm_stats", {}))
    for r in range(state.get("rounds", 1)):
        try:
            crit, st_c = call_json(
                CRITIQUE_SYSTEM,
                CRITIQUE_USER.format(
                    criteria=CRITERIA_TEXT,
                    arguments=_format_args(current["arguments"], sentences),
                    document=state["agent_text"],
                ),
                CritiqueOutput,
                **_llm(state),
                temperature=0.0,
                num_predict=1500,
            )
            critiques = crit.model_dump()["critiques"]
            refined, st_r = call_json(
                REFINE_SYSTEM,
                REFINE_USER.format(
                    criteria=CRITERIA_TEXT,
                    arguments=_format_args(current["arguments"], sentences, critiques),
                    document=state["agent_text"],
                ),
                DebateOutput,
                **_llm(state),
                temperature=0.3,
                num_predict=1800,
            )
        except LLMOutputError as exc:
            errors.append(f"critique/raffinement tour {r + 1} : {exc}")
            break
        history.append({"arguments": current["arguments"], "critiques": critiques})
        stats[f"critique_{r + 1}"], stats[f"refine_{r + 1}"] = st_c, st_r
        current = refined.model_dump()
    return {
        "initial_debate": state["debate"],
        "debate": current,
        "rounds_history": history,
        "errors": errors,
        "llm_stats": stats,
    }


def judge_anchored(state: CommitteeState) -> dict:
    if not state.get("debate"):
        return {"judge": None}
    try:
        out, stats = call_json(
            JUDGE_SYSTEM,
            JUDGE_ANCHORED_USER.format(
                criteria=CRITERIA_TEXT,
                rubric=ANCHORED_RUBRIC,
                arguments=_format_args(state["debate"]["arguments"], state.get("sentences", [])),
            ),
            JudgeOutput,
            **_llm(state),
            temperature=0.0,
            num_predict=1200,
        )
    except LLMOutputError as exc:
        return {"judge": None, "errors": [*state.get("errors", []), f"juge : {exc}"]}
    return {"judge": out.model_dump(), "llm_stats": {**state.get("llm_stats", {}), "judge": stats}}


def build_full_committee():
    """Comité complet du bloc C (le nombre de tours est lu dans l'état : `rounds`, défaut 1)."""
    g = StateGraph(CommitteeState)
    g.add_node("extract", extract)
    g.add_node("debate", debate)
    g.add_node("critique_refine", critique_refine)
    g.add_node("judge", judge_anchored)
    g.add_node("verify", verify)
    g.add_edge(START, "extract")
    g.add_edge("extract", "debate")
    g.add_edge("debate", "critique_refine")
    g.add_edge("critique_refine", "judge")
    g.add_edge("judge", "verify")
    g.add_edge("verify", END)
    return g.compile()


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
