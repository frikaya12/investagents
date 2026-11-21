"""Schémas Pydantic des sorties du comité (validés à chaque appel LLM).

Structure inspirée de `Argument` de DIALECTIC (github.com/pantageepapa/DIALECTIC, MIT), adaptée :
chaque argument est rattaché à un des 5 critères et porte une citation exacte du document.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CRITERIA = (
    "relevance",
    "economic_efficiency",
    "feasibility",
    "risk",
    "impact_sustainability",
)
Criterion = Literal[
    "relevance", "economic_efficiency", "feasibility", "risk", "impact_sustainability"
]


class DebateArgument(BaseModel):
    criterion: Criterion
    stance: Literal["pro", "contra"]
    claim: str = Field(min_length=10, max_length=600)
    sentence_ids: list[int] = Field(
        min_length=1, max_length=3, description="Numéros des phrases citées (S1 = 1)"
    )


class DebateOutput(BaseModel):
    arguments: list[DebateArgument] = Field(min_length=5, max_length=20)

    @field_validator("arguments")
    @classmethod
    def every_criterion_has_pro_and_contra(cls, args):
        seen = {(a.criterion, a.stance) for a in args}
        missing = [(c, s) for c in CRITERIA for s in ("pro", "contra") if (c, s) not in seen]
        if missing:
            raise ValueError(f"Arguments manquants : {missing}")
        return args


class CriterionScore(BaseModel):
    criterion: Criterion
    score: int = Field(ge=0, le=10)
    justification: str = Field(min_length=10, max_length=600)


class JudgeOutput(BaseModel):
    scores: list[CriterionScore] = Field(min_length=5, max_length=5)

    @field_validator("scores")
    @classmethod
    def one_score_per_criterion(cls, scores):
        if sorted(s.criterion for s in scores) != sorted(CRITERIA):
            raise ValueError("Il faut exactement une note par critère")
        return scores


# ---------------------------------------------------------------- comité complet (bloc C)
class CritiqueItem(BaseModel):
    """Critique d'UN argument (repris de l'évaluation d'arguments de DIALECTIC : acceptabilité,
    pertinence, suffisance), sous forme de verdict qualitatif, sans chiffre."""

    argument_index: int = Field(ge=0, le=19)
    verdict: Literal["strong", "weak", "unsupported"]
    comment: str = Field(min_length=5, max_length=400)


class CritiqueOutput(BaseModel):
    critiques: list[CritiqueItem] = Field(min_length=5, max_length=20)


class RecognitionOutput(BaseModel):
    """Test de reconnaissance : le modèle connaît-il déjà ce projet (mémorisation) ?
    Les réponses sont comparées à la vérité EN LOCAL (la vérité n'est jamais envoyée)."""

    recognized: bool
    project_name_guess: str = Field(max_length=200)
    country_guess: str = Field(max_length=100)
    approval_year_guess: int | None = None
    outcome_guess: Literal[
        "Highly Satisfactory",
        "Satisfactory",
        "Moderately Satisfactory",
        "Moderately Unsatisfactory",
        "Unsatisfactory",
        "Highly Unsatisfactory",
        "unknown",
    ]
