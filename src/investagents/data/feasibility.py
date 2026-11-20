"""Étude de faisabilité (phase 1) : combien de projets notés ont un document ex ante en texte ?

Deux mesures complémentaires :
1. Échantillon : 50 projets notés tirés au hasard (graine 42). Pour chacun, on applique le filtre
   anti-fuite à TOUS ses documents, puis on télécharge réellement le texte des documents gardés
   et on mesure sa longueur. On extrapole avec un intervalle de Wilson à 95 %.
2. Recensement : on liste via l'API TOUS les dossiers d'évaluation (tier 1) de la Banque
   mondiale, on applique le même filtre à tous les projets notés, et on compte ceux qui ont
   un dossier avec un `txturl`. Le texte n'est pas téléchargé pour tous (des milliers de
   fichiers) : la part de textes inutilisables est estimée sur l'échantillon.

Lancer : `uv run python -m investagents.data.feasibility`  (écrit reports/faisabilite_chiffres.md)
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from investagents.data import documents_api as da
from investagents.data.leakage import TIER1_TYPES, filter_ex_ante

# Seuil fixé AVANT de regarder les longueurs : un dossier d'évaluation complet fait des
# dizaines de pages (le PAD de P122764 : 416 047 caractères). En dessous de 10 000 caractères
# (environ 5 pages), le fichier texte est vide, tronqué ou un OCR raté.
MIN_CHARS = 10_000

CENSUS_DOCS_PATH = Path("data/processed/census_tier1_documents.parquet")
SAMPLE_STATUS_PATH = Path("data/processed/sample50_status.parquet")
REPORT_PATH = Path("reports/faisabilite_chiffres.md")

STATUS_LABELS = {
    "A_tier1_text": "Dossier d'évaluation (tier 1) avec texte exploitable",
    "B_tier2_text": "Seulement un document ex ante court (tier 2) avec texte exploitable",
    "C_ex_ante_no_text": "Document ex ante présent mais texte absent ou inutilisable",
    "D_no_ex_ante": "Aucun document ex ante daté avant l'approbation",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion k/n (meilleur que p ± 1,96·se
    quand n est petit ou p proche de 0 ou 1)."""
    if n <= 0:
        raise ValueError("n doit être positif")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def build_approvals(ieg: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    """Table project_id / approval_date (Projects API) / approval_fy (IEG)."""
    appr = ieg[["project_id", "approval_fy"]].merge(
        projects[["project_id", "approval_date"]], on="project_id", how="left"
    )
    return appr


def classify_projects(
    project_ids: list[str], kept: pd.DataFrame, min_chars: int = MIN_CHARS
) -> pd.DataFrame:
    """Statut A/B/C/D de chaque projet, à partir des documents gardés et de `n_chars`."""
    rows = []
    for pid in project_ids:
        d = kept[kept["project_id"] == pid]
        ok = d[d["n_chars"] >= min_chars]
        if (ok["tier"] == 1).any():
            status = "A_tier1_text"
        elif (ok["tier"] == 2).any():
            status = "B_tier2_text"
        elif len(d):
            status = "C_ex_ante_no_text"
        else:
            status = "D_no_ex_ante"
        rows.append(
            {
                "project_id": pid,
                "status": status,
                "n_ex_ante_docs": len(d),
                "max_chars": int(d["n_chars"].max()) if len(d) else 0,
            }
        )
    return pd.DataFrame(rows)


def fetch_census(get=None) -> pd.DataFrame:
    """Tous les documents des types tier 1, toutes années confondues."""
    frames = []
    for docty in sorted(TIER1_TYPES):
        kwargs = {} if get is None else {"get": get}
        docs = da.fetch_documents({"docty_exact": docty}, **kwargs)
        print(f"Recensement : {docty} -> {len(docs)} documents")
        frames.append(da.parse_documents(docs))
    return pd.concat(frames, ignore_index=True)


def _period(fy: pd.Series) -> pd.Series:
    return pd.cut(
        fy.astype(float),
        bins=[0, 1997, 2007, 2015, 9999],
        labels=["≤ 1997", "1998–2007", "2008–2015", "≥ 2016"],
    )


def main() -> None:
    ieg = pd.read_parquet("data/processed/ieg_ratings.parquet")
    projects = pd.read_parquet("data/processed/projects.parquet")
    rated = ieg[ieg["satisfactory"].notna()].copy()
    approvals = build_approvals(rated, projects)
    n_rated = len(rated)

    # ---------- 1. Échantillon de 50 projets ----------
    sample = pd.read_parquet(da.SAMPLE_PROJECTS_PATH)
    docs = pd.read_parquet(da.SAMPLE_PATH)
    kept = filter_ex_ante(docs, approvals)
    kept["n_chars"] = [
        da.download_text(url, doc_id)
        for url, doc_id in zip(kept["txturl"], kept["doc_id"], strict=True)
    ]
    status = classify_projects(sample["project_id"].tolist(), kept)
    status = status.merge(
        sample[["project_id", "approval_fy", "outcome", "satisfactory"]], on="project_id"
    ).merge(approvals[["project_id", "approval_date"]], on="project_id", how="left")
    status.to_parquet(SAMPLE_STATUS_PATH, index=False)

    n = len(status)
    counts = status["status"].value_counts().reindex(STATUS_LABELS, fill_value=0)
    k_a = int(counts["A_tier1_text"])
    k_ab = k_a + int(counts["B_tier2_text"])
    ci_a = wilson_ci(k_a, n)
    ci_ab = wilson_ci(k_ab, n)

    tier1_kept = kept[kept["tier"] == 1]
    tier1_with_url = tier1_kept[tier1_kept["txturl"].notna()]
    n_url_ok = int((tier1_with_url["n_chars"] >= MIN_CHARS).sum())

    # ---------- 2. Recensement des dossiers d'évaluation ----------
    if CENSUS_DOCS_PATH.exists():
        census_docs = pd.read_parquet(CENSUS_DOCS_PATH)
        print(f"Cache utilisé : {CENSUS_DOCS_PATH}")
    else:
        census_docs = fetch_census()
        census_docs.to_parquet(CENSUS_DOCS_PATH, index=False)
    census_kept = filter_ex_ante(census_docs, approvals)
    census_kept = census_kept[census_kept["txturl"].notna()]
    census_pids = set(census_kept["project_id"])
    rated["census_tier1"] = rated["project_id"].isin(census_pids)
    n_census = int(rated["census_tier1"].sum())
    # Documents tier 1 rejetés uniquement à cause de la date (pour mesurer l'effet du filtre).
    tier1_rated = census_docs[census_docs["project_id"].isin(set(rated["project_id"]))]
    same_day = tier1_rated.merge(approvals, on="project_id")
    n_same_day = int((same_day["docdt"] == same_day["approval_date"]).sum())
    n_after = int((same_day["docdt"] > same_day["approval_date"]).sum())

    # Sensibilité (information seulement, la règle stricte reste celle de leakage.py) :
    # combien de projets en plus si l'on acceptait un dossier daté ≤ 31 jours après
    # l'approbation (dates au mois près des vieux SAR, versions finales publiées après le Conseil).
    late_ok = same_day[
        (same_day["lang"] == "English")
        & same_day["txturl"].notna()
        & (same_day["docdt"] >= same_day["approval_date"])
        & ((same_day["docdt"] - same_day["approval_date"]).dt.days <= 31)
    ]
    n_tolerance_extra = len(set(late_ok["project_id"]) - census_pids)
    no_date = rated.merge(approvals[["project_id", "approval_date"]], on="project_id")
    n_no_approval_date = int(no_date["approval_date"].isna().sum())

    ex = rated[rated["census_tier1"]]
    by_period = (
        pd.DataFrame(
            {
                "notés": rated.groupby(_period(rated["approval_fy"]), observed=False).size(),
                "avec dossier tier 1 (txturl)": ex.groupby(
                    _period(ex["approval_fy"]), observed=False
                ).size(),
            }
        )
        .rename_axis("Exercice d'approbation")
        .reset_index()
    )
    by_period["part"] = (by_period.iloc[:, 2] / by_period.iloc[:, 1]).map("{:.0%}".format)
    by_outcome = ex["outcome"].value_counts()
    src = census_kept.drop_duplicates("project_id")["approval_ref_source"].value_counts()

    # ---------- 3. Fichier de chiffres ----------
    lines = [
        "# Faisabilité : chiffres bruts (fichier généré automatiquement)",
        "",
        "Généré par `uv run python -m investagents.data.feasibility`. Ne pas éditer à la main.",
        "",
        f"- Projets IEG notés (cible binaire connue) : **{n_rated}**",
        f"- Seuil de texte exploitable : {MIN_CHARS:,} caractères".replace(",", " "),
        "",
        "## 1. Échantillon de 50 projets notés (graine 42)",
        "",
        "| Statut | Projets |",
        "| --- | ---: |",
        *[f"| {STATUS_LABELS[s]} | {int(c)} |" for s, c in counts.items()],
        f"| **Total** | **{n}** |",
        "",
        f"- Part avec dossier d'évaluation en texte (A) : {k_a}/{n} = {k_a / n:.0%}, "
        f"IC 95 % Wilson [{ci_a[0]:.0%} ; {ci_a[1]:.0%}]",
        f"- Extrapolation à {n_rated} projets notés : {k_a / n * n_rated:,.0f} projets "
        f"[{ci_a[0] * n_rated:,.0f} ; {ci_a[1] * n_rated:,.0f}]".replace(",", " "),
        f"- Avec le repli tier 2 (A + B) : {k_ab}/{n} = {k_ab / n:.0%}, "
        f"IC 95 % [{ci_ab[0]:.0%} ; {ci_ab[1]:.0%}] -> {k_ab / n * n_rated:,.0f} projets "
        f"[{ci_ab[0] * n_rated:,.0f} ; {ci_ab[1] * n_rated:,.0f}]".replace(",", " "),
        f"- Documents tier 1 gardés avec un txturl : {len(tier1_with_url)} ; "
        f"dont texte ≥ seuil après téléchargement : {n_url_ok}",
        "",
        "Détail par projet :",
        "",
        "| Projet | Exercice | Note IEG | Statut | Docs ex ante | Longueur max (car.) |",
        "| --- | ---: | --- | --- | ---: | ---: |",
        *[
            f"| {r.project_id} | {r.approval_fy} | {r.outcome} | {r.status} | "
            f"{r.n_ex_ante_docs} | {r.max_chars:,} |".replace(",", " ")
            for r in status.sort_values("approval_fy").itertuples()
        ],
        "",
        "## 2. Recensement complet des dossiers d'évaluation (tier 1)",
        "",
        f"- Documents tier 1 listés par l'API : {len(census_docs)} lignes document-projet",
        f"- Projets notés avec ≥ 1 dossier tier 1 en anglais, daté avant l'approbation, "
        f"avec txturl : **{n_census}** ({n_census / n_rated:.0%} des notés)",
        f"- Référence de date utilisée : {src.to_dict()}",
        f"- Dossiers tier 1 de projets notés datés le jour même de l'approbation (rejetés) : "
        f"{n_same_day} ; datés après (rejetés) : {n_after}",
        f"- Sensibilité : projets en plus si l'on tolérait un dossier daté ≤ 31 jours après "
        f"l'approbation (NON appliqué) : {n_tolerance_extra}",
        f"- Projets notés sans date d'approbation exacte dans la Projects API : "
        f"{n_no_approval_date}",
        f"- Estimation après contrôle du texte (taux de l'échantillon {n_url_ok}/"
        f"{len(tier1_with_url)}) : {n_census * n_url_ok / max(1, len(tier1_with_url)):,.0f} "
        f"projets".replace(",", " "),
        "",
        by_period.to_markdown(index=False),
        "",
        "Répartition des notes parmi ces projets :",
        "",
        by_outcome.rename_axis("Note IEG").reset_index(name="projets").to_markdown(index=False),
        "",
        f"- Non satisfaisants (MU/U/HU) : {int((ex['satisfactory'] == 0).sum())} ; "
        f"satisfaisants : {int((ex['satisfactory'] == 1).sum())}",
        f"- Dont évalués par l'IEG en exercice ≥ 2025 : {int((ex['evaluation_fy'] >= 2025).sum())}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
