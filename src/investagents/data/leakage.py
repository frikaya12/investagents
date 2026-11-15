"""Filtre anti-fuite : seuls les documents ex ante datés avant l'approbation passent.

C'est le SEUL endroit du projet où l'on décide quels documents un agent a le droit de lire
(règle n° 1 de CLAUDE.md). Le principe : liste blanche + date stricte + rejet en cas de doute.

Un document est gardé si et seulement si :
1. son type est dans une liste blanche (TIER1_TYPES, TIER2_TYPES, ou une fiche d'information
   dont le titre contient « Appraisal Stage ») ;
2. son type n'est pas dans la liste noire FORBIDDEN_TYPES et son titre ne contient aucun mot
   suspect (« Completion », « Restructuring »...) : double sécurité ;
3. il est en anglais (les traductions existent ; les agents lisent l'anglais) ;
4. sa date `docdt` est connue et STRICTEMENT antérieure à la date de référence :
   - la date d'approbation par le Conseil (Projects API) si elle est connue ;
   - sinon le 1er juillet qui ouvre l'exercice d'approbation IEG (borne prudente : un document
     daté avant ce jour est forcément antérieur à l'approbation) ;
   - sinon le document est rejeté.

Niveaux : tier 1 = dossier d'évaluation (PAD, SAR, Program Document) ; tier 2 = autre document
ex ante plus court (mémorandum du Président, fiche d'information au stade de l'évaluation).
"""

from __future__ import annotations

import pandas as pd

# Types confirmés dans l'API Documents (valeurs exactes de `docty`, vérifiées le 2026-11-15).
TIER1_TYPES = frozenset(
    {
        "Project Appraisal Document",  # PAD : dossier d'évaluation (depuis la fin des années 1990)
        "Staff Appraisal Report",  # SAR : ancien nom du dossier d'évaluation
        "Program Document",  # dossier d'évaluation des prêts à l'appui des politiques (DPF)
    }
)
TIER2_TYPES = frozenset(
    {
        "Memorandum & Recommendation of the President",  # présentation du prêt au Conseil
        "President's Report",  # idem, projets anciens
    }
)
# Fiches d'information : autorisées seulement si le titre dit « Appraisal Stage ».
APPRAISAL_STAGE_TYPES = frozenset({"Project Information Document", "Program Information Document"})
APPRAISAL_STAGE_MARKER = "appraisal stage"

# Liste noire : documents qui contiennent (ou peuvent contenir) l'issue du projet.
FORBIDDEN_TYPES = frozenset(
    {
        "Implementation Completion and Results Report",  # ICR
        "Implementation Completion Report Review",  # ICRR (revue IEG)
        "Implementation Completion Memorandum",
        "Project Performance Assessment Report",  # PPAR (IEG)
        "Project Completion Report",  # PCR (ancien ICR)
        "Implementation Status and Results Report",  # ISR (suivi en cours de projet)
        "Evaluation Memorandum",
        "IEG Evaluation",
        "Project Paper",  # restructuration / financement additionnel, après approbation
        "Aide Memoire",
        "Auditing Document",
        "Procurement Plan",
    }
)
SUSPICIOUS_TITLE_WORDS = (
    "completion",
    "restructuring",
    "performance assessment",
    "additional financing",
    "results report",
)
ALLOWED_LANG = "English"


def fiscal_year_start(fy: int) -> pd.Timestamp:
    """Exercice Banque mondiale N = 1er juillet N-1 -> 30 juin N."""
    return pd.Timestamp(year=int(fy) - 1, month=7, day=1)


def document_tier(docty: str | None, title: str | None) -> int | None:
    """1 ou 2 si le type est autorisé, None sinon (titre suspect ou type non autorisé)."""
    if not docty or docty in FORBIDDEN_TYPES:
        return None
    low_title = (title or "").lower()
    if any(word in low_title for word in SUSPICIOUS_TITLE_WORDS):
        return None
    if docty in TIER1_TYPES:
        return 1
    if docty in TIER2_TYPES:
        return 2
    if docty in APPRAISAL_STAGE_TYPES and APPRAISAL_STAGE_MARKER in low_title:
        return 2
    return None


def approval_reference(approvals: pd.DataFrame) -> pd.DataFrame:
    """Date de référence par projet : date d'approbation, sinon début de l'exercice IEG."""
    ref = approvals[["project_id", "approval_date", "approval_fy"]].copy()
    ref["approval_date"] = pd.to_datetime(ref["approval_date"])
    fy_start = ref["approval_fy"].map(lambda fy: fiscal_year_start(fy) if pd.notna(fy) else pd.NaT)
    has_date = ref["approval_date"].notna()
    ref["approval_ref"] = ref["approval_date"].where(has_date, pd.to_datetime(fy_start))
    ref["approval_ref_source"] = "board_approval_date"
    ref.loc[~has_date, "approval_ref_source"] = "fiscal_year_start"
    ref.loc[ref["approval_ref"].isna(), "approval_ref_source"] = None
    return ref[["project_id", "approval_ref", "approval_ref_source"]]


def filter_ex_ante(docs: pd.DataFrame, approvals: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les documents ex ante autorisés, datés strictement avant la référence.

    `docs` : colonnes project_id, docty, docdt, display_title, lang (sortie de documents_api).
    `approvals` : colonnes project_id, approval_date (Projects API), approval_fy (IEG).
    Renvoie les documents gardés avec `tier`, `approval_ref`, `approval_ref_source`.
    """
    out = docs.merge(approval_reference(approvals), on="project_id", how="inner")
    out["tier"] = [
        document_tier(t, title) for t, title in zip(out["docty"], out["display_title"], strict=True)
    ]
    keep = (
        out["tier"].notna()
        & (out["lang"] == ALLOWED_LANG)
        & out["docdt"].notna()
        & out["approval_ref"].notna()
        & (out["docdt"] < out["approval_ref"])
    )
    out = out[keep].copy()
    out["tier"] = out["tier"].astype(int)
    return out.reset_index(drop=True)
