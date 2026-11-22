"""Bloc D : grille FIGÉE de 19 facteurs de risque factuels lus dans le dossier ex ante.

Chaque facteur est une question factuelle (pas un jugement) posée à un agent extracteur, qui doit
citer les numéros des phrases qui le justifient, ou répondre « not_found ». Trois types :
- "yes_no"   : value ∈ {yes, no} ;
- "category" : value ∈ liste fermée de catégories ;
- "list"     : items = noms courts (composantes, agences, cofinanceurs) ; le NOMBRE est compté en
               Python (règle : aucun chiffre ne vient du LLM).
`keywords` (expression régulière, insensible à la casse) sert à présélectionner en Python les
phrases candidates du document complet, et à contrôler que les phrases citées en parlent.

La grille est figée AVANT le pilote : son empreinte SHA-256 est écrite dans
experiments/blocD/factor_grid.json (`uv run python -m investagents.extract.factors`), et un test
vérifie que la grille du code correspond à ce fichier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

GRID_PATH = Path("experiments/blocD/factor_grid.json")


@dataclass(frozen=True)
class Factor:
    name: str
    kind: str  # yes_no | category | list
    question: str
    keywords: str
    categories: tuple[str, ...] = field(default_factory=tuple)


FACTORS: tuple[Factor, ...] = (
    Factor("components", "list",
           "List the project components (short name of each component).",
           r"\bcomponent"),
    Factor("implementing_agencies", "list",
           "List the agencies or entities responsible for implementing the project.",
           r"implementing agenc|implementing entit|executing agenc|project implementation unit"
           r"|\bPIU\b|\bPMU\b|implementation unit|responsible for implementation"),
    Factor("cofinanciers", "list",
           "List the other financiers or donors co-financing the project (not the Bank, not "
           "the government).",
           r"co-?financ|parallel financ|donors?\b|trust fund|grant from"),
    Factor("first_operation", "yes_no",
           "Is this described as the first Bank operation in this sector, with this agency, "
           "or in this country?",
           r"first (?:bank|ida|world bank)?\s*(?:operation|project|engagement|loan|credit)"
           r"|first time|no previous|new to the bank"),
    Factor("follow_on_phase", "yes_no",
           "Is the project a second (or later) phase, a follow-on, or part of an APL series?",
           r"second phase|phase (?:2|ii|two)|follow-on|follow up|adaptable program|\bAPL\b"
           r"|repeater|previous phase"),
    Factor("reform_to_adopt", "yes_no",
           "Does the project depend on a law, regulation, decree or reform that must still be "
           "adopted (e.g. a condition of effectiveness or disbursement)?",
           r"\blaw\b|legislation|decree|regulation|condition of effectiveness"
           r"|effectiveness condition|disbursement condition|prior action|adoption of"),
    Factor("env_category", "category",
           "What environmental assessment category is the project assigned?",
           r"environmental (?:assessment |screening )?category|category [abc]\b"
           r"|\bEA category|categor(?:y|ized) as",
           ("A", "B", "C", "FI")),
    Factor("overall_risk_rating", "category",
           "What overall risk rating does the document give the project?",
           r"overall risk|risk rating|rated (?:as )?(?:high|substantial|moderate|low|modest)"
           r"|\bSORT\b",
           ("low", "moderate", "substantial", "high")),
    Factor("fiduciary_risk", "category",
           "What risk rating is given to financial management or procurement?",
           r"financial management risk|procurement risk|fiduciary risk"
           r"|(?:financial management|procurement)[^.]{0,60}(?:rated|risk is|assessed)",
           ("low", "moderate", "substantial", "high")),
    Factor("baseline_values", "yes_no",
           "Do the results / outcome indicators have baseline values stated?",
           r"baseline|base year|indicator"),
    Factor("lessons_problems", "yes_no",
           "Do the lessons learned mention a previous project that performed poorly, failed, or "
           "was rated unsatisfactory?",
           r"lessons?|unsatisfactory|poor performance|did not achieve|failed|shortcoming"),
    Factor("fragility_conflict", "yes_no",
           "Is the country or project area described as fragile, in conflict, or post-conflict?",
           r"fragil|conflict|post-conflict|civil war|insecurity|violence|refugee"),
    Factor("weak_capacity", "yes_no",
           "Does the document state that the implementing agency has weak or limited capacity?",
           r"weak capacity|limited capacity|capacity constraint|lack of capacity"
           r"|capacity is (?:weak|limited)|insufficient capacity|capacity weakness"),
    Factor("counterpart_funding", "yes_no",
           "Is government counterpart funding required for the project?",
           r"counterpart fund|government contribution|counterpart contribution"
           r"|borrower contribution|co-financing by the government"),
    Factor("resettlement", "yes_no",
           "Is the involuntary resettlement safeguard triggered, or is land acquisition or "
           "resettlement expected?",
           r"resettlement|land acquisition|OP 4\.12|OP/BP 4\.12|displac"),
    Factor("decentralized_implementation", "yes_no",
           "Is implementation decentralized to many local or sub-national entities "
           "(municipalities, provinces, communities)?",
           r"municipalit|provinc|local government|communit|decentrali|sub-national|district"),
    Factor("pilot_innovative", "yes_no",
           "Is the project described as a pilot or as an innovative approach new to the country?",
           r"\bpilot|innovative|new approach|first of its kind|demonstration"),
    Factor("political_risk", "yes_no",
           "Does the document list political risk, elections, or weak government commitment "
           "as a risk?",
           r"political|election|government commitment|ownership|political economy"),
    Factor("economic_return_reported", "yes_no",
           "Does the document report an economic rate of return, net present value or "
           "cost-benefit result?",
           r"economic rate of return|\bERR\b|\bEIRR\b|net present value|\bNPV\b"
           r"|cost-benefit|benefit-cost|internal rate of return"),
)  # fmt: skip

NAMES = tuple(f.name for f in FACTORS)


def grid_dict() -> dict:
    factors = [asdict(f) for f in FACTORS]
    for f in factors:
        f["categories"] = list(f["categories"])
    return {"version": 1, "n_factors": len(factors), "factors": factors}


def grid_hash() -> str:
    blob = json.dumps(grid_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {**grid_dict(), "sha256": grid_hash()}
    GRID_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Grille figée : {len(FACTORS)} facteurs, sha256 {out['sha256'][:12]}… → {GRID_PATH}")


if __name__ == "__main__":
    main()
