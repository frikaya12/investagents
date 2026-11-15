"""Tests du filtre anti-fuite (écrits avant le code de leakage.py).

Règle : un agent ne lit qu'un document ex ante (type autorisé) daté STRICTEMENT avant
l'approbation du projet. Tout le reste est rejeté, y compris en cas de doute (date manquante).
"""

import pandas as pd
import pytest

from investagents.data import leakage as lk

APPROVAL = pd.Timestamp("2012-05-24")


def doc(docty, docdt, title="Some Project", project_id="P122764", lang="English"):
    return {
        "doc_id": f"D-{docty[:6]}-{docdt}",
        "project_id": project_id,
        "docty": docty,
        "docdt": pd.Timestamp(docdt) if docdt else pd.NaT,
        "display_title": title,
        "lang": lang,
        "txturl": "http://x.txt",
    }


def approvals(date=APPROVAL, fy=2012, project_id="P122764"):
    return pd.DataFrame({"project_id": [project_id], "approval_date": [date], "approval_fy": [fy]})


def run(docs, appr=None):
    return lk.filter_ex_ante(pd.DataFrame(docs), approvals() if appr is None else appr)


# --- Règle de date ---------------------------------------------------------------------------


def test_keeps_pad_dated_before_approval():
    out = run([doc("Project Appraisal Document", "2012-04-26")])
    assert len(out) == 1
    assert out.iloc[0]["tier"] == 1


def test_rejects_fake_pad_dated_after_approval():
    """Faux document : un « PAD » daté après l'approbation doit être rejeté."""
    out = run([doc("Project Appraisal Document", "2013-01-15")])
    assert out.empty


def test_rejects_document_dated_on_approval_day():
    out = run([doc("Project Appraisal Document", "2012-05-24")])
    assert out.empty


def test_rejects_document_without_date():
    out = run([doc("Project Appraisal Document", None)])
    assert out.empty


def test_rejects_when_approval_unknown_and_no_fiscal_year():
    appr = approvals(date=pd.NaT, fy=pd.NA)
    out = run([doc("Project Appraisal Document", "2000-01-01")], appr)
    assert out.empty


def test_rejects_project_absent_from_approval_table():
    out = run([doc("Project Appraisal Document", "2012-04-26", project_id="P999999")])
    assert out.empty


def test_fiscal_year_fallback_is_conservative():
    """Sans date exacte, la borne est le 1er juillet qui ouvre l'exercice d'approbation."""
    appr = approvals(date=pd.NaT, fy=2012)  # exercice 2012 = 1er juillet 2011 -> 30 juin 2012
    out = run(
        [
            doc("Project Appraisal Document", "2011-06-30"),  # avant le début de l'exercice : OK
            doc("Staff Appraisal Report", "2011-09-01"),  # dans l'exercice : date incertaine
        ],
        appr,
    )
    assert out["docty"].tolist() == ["Project Appraisal Document"]
    assert out.iloc[0]["approval_ref_source"] == "fiscal_year_start"


def test_fiscal_year_start_helper():
    assert lk.fiscal_year_start(2012) == pd.Timestamp("2011-07-01")


# --- Règle de type ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "docty",
    [
        "Implementation Completion and Results Report",  # ICR
        "Implementation Completion Report Review",  # ICRR
        "Project Performance Assessment Report",  # PPAR
        "Project Completion Report",
        "Implementation Status and Results Report",
        "Evaluation Memorandum",
        "IEG Evaluation",
        "Project Paper",
        "Procurement Plan",
    ],
)
def test_rejects_forbidden_types_even_if_dated_before(docty):
    out = run([doc(docty, "2010-01-01")])
    assert out.empty


def test_forbidden_and_allowed_lists_do_not_overlap():
    assert not (lk.TIER1_TYPES | lk.TIER2_TYPES | lk.APPRAISAL_STAGE_TYPES) & lk.FORBIDDEN_TYPES


def test_tiers():
    out = run(
        [
            doc("Staff Appraisal Report", "1990-01-01"),
            doc("Program Document", "2011-01-01"),
            doc("Memorandum & Recommendation of the President", "2011-02-01"),
            doc("President's Report", "1980-01-01"),
        ]
    )
    tiers = dict(zip(out["docty"], out["tier"], strict=True))
    assert tiers["Staff Appraisal Report"] == 1
    assert tiers["Program Document"] == 1
    assert tiers["Memorandum & Recommendation of the President"] == 2
    assert tiers["President's Report"] == 2


def test_pid_only_at_appraisal_stage():
    out = run(
        [
            doc("Project Information Document", "2012-02-14", title="PID (Appraisal Stage) - X"),
            doc("Project Information Document", "2010-10-25", title="Ethiopia - X Project"),
            doc("Program Information Document", "2011-01-01", title="Concept Stage - X"),
        ]
    )
    assert len(out) == 1
    assert "Appraisal Stage" in out.iloc[0]["display_title"]
    assert out.iloc[0]["tier"] == 2


@pytest.mark.parametrize(
    "title",
    [
        "Implementation Completion and Results Report - X",
        "Restructuring Paper - X",
        "Project Performance Assessment Report - X",
    ],
)
def test_rejects_suspicious_titles_even_with_allowed_type(title):
    out = run([doc("Project Appraisal Document", "2012-04-26", title=title)])
    assert out.empty


def test_rejects_non_english_versions():
    out = run([doc("Project Appraisal Document", "2012-04-26", lang="Spanish")])
    assert out.empty


def test_output_never_contains_forbidden_or_late_documents():
    """Propriété globale sur un mélange réaliste."""
    docs = [
        doc("Project Appraisal Document", "2012-04-26"),
        doc("Project Appraisal Document", "2014-04-26"),
        doc("Implementation Completion and Results Report", "2011-01-01"),
        doc("Implementation Completion Report Review", "2016-01-01"),
        doc("Project Information Document", "2012-02-14", title="PID (Appraisal Stage)"),
        doc("Integrated Safeguards Data Sheet", "2012-02-08"),
    ]
    out = run(docs)
    assert (out["docdt"] < APPROVAL).all()
    assert not out["docty"].isin(lk.FORBIDDEN_TYPES).any()
    assert len(out) == 2
