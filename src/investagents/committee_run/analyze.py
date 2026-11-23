"""Analyse du passage du comité sur Kaggle (bloc C) — sans M3 / M4 et sans regarder l'issue du test.

Lit experiments/blocC/kaggle_run1/{results,recognition}.jsonl et la correspondance locale
experiments/blocC/sample.csv. Écrit reports/blocC_comite.md et experiments/blocC/scores.parquet.

1. Fonctionnement : projets notés, erreurs, temps par projet, coût.
2. Citations : (a) numéros existants (vérifié sur Kaggle) ; (b) vérification mot pour mot refaite
   ici : chaque phrase citée doit se retrouver dans le texte complet masqué du dossier.
3. Dispersion des notes (demande d'Aya) : moyenne, écart-type par critère, part des notes ≥ 9 et
   = 10 — toutes périodes (descriptif, n'utilise pas l'issue).
4. Critique et raffinement : verdicts du critique, part des arguments modifiés.
5. Test de reconnaissance (50 projets) : réponses comparées à la vérité EN LOCAL.
6. Lien notes / issue sur la VALIDATION seulement (AUC par critère, diagnostic). Le test n'est
   pas touché : M3 / M4 attendent la validation d'Aya.

Lancer : `uv run python -m investagents.committee_run.analyze`
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from investagents.agents.citations import quote_found
from investagents.agents.sections import mask_identifiers, normalize
from investagents.committee_run.runner import load_jsonl
from investagents.data.corpus import read_text
from investagents.schemas.committee import CRITERIA

RUN = Path("experiments/blocC/kaggle_run1")
SAMPLE = Path("experiments/blocC/sample.csv")
REPORT = Path("reports/blocC_comite.md")
SCORES = Path("experiments/blocC/scores.parquet")
SUMMARY = Path("experiments/blocC/committee_summary.json")  # chiffres repris par le rapport final


def full_text_check(results: list[dict], sample: pd.DataFrame) -> tuple[int, int]:
    """(citations valides mot pour mot, citations) sur le texte complet masqué."""
    meta = sample.set_index("key")
    ok = total = 0
    for r in results:
        if not r.get("citations"):
            continue
        m = meta.loc[r["key"]]
        full = mask_identifiers(normalize(read_text(m.doc_id)), name=m.project_name)
        for c in r["citations"]:
            total += 1
            ok += bool(c["valid"]) and all(quote_found(t, full) for t in c["cited_text"])
    return ok, total


def recognition_table(rec: list[dict], sample: pd.DataFrame) -> pd.DataFrame:
    meta = sample.set_index("key")
    rows = []
    for r in rec:
        if "error" in r:
            continue
        m = meta.loc[r["key"]]
        guess, truth = str(r["country_guess"]).lower(), str(m.country).lower()
        name_sim = difflib.SequenceMatcher(
            None, str(r["project_name_guess"]).lower(), str(m.project_name).lower()
        ).ratio()
        year = r.get("approval_year_guess")
        outcome = r["outcome_guess"]
        rows.append(
            {
                "key": r["key"],
                "split": r["split"],
                "recognized": r["recognized"],
                "country_ok": bool(guess) and guess not in ("unknown", "")
                and (guess in truth or truth in guess),
                "year_within_1": year is not None and abs(int(year) - int(m.approval_fy)) <= 1,
                "name_similarity": name_sim,
                "outcome_guess": outcome,
                "outcome_known": outcome != "unknown",
                "outcome_guess_fail": outcome.startswith(("Moderately Uns", "Uns", "Highly Uns")),
                "y_fail": int(m.y_fail),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    results = load_jsonl(RUN / "results.jsonl")
    results = list({r["key"]: r for r in results}.values())  # dernière version par projet
    rec = load_jsonl(RUN / "recognition.jsonl")
    prog = json.loads((RUN / "progress.json").read_text(encoding="utf-8"))

    rows = []
    for r in results:
        row = {"key": r["key"], "seconds": r["seconds"], "n_errors": len(r.get("errors", [])),
               "citation_valid_rate": r.get("citation_valid_rate")}  # fmt: skip
        if r.get("judge"):
            row.update({s["criterion"]: s["score"] for s in r["judge"]["scores"]})
        verdicts = [c["verdict"] for h in r.get("rounds_history", []) for c in h["critiques"]]
        row["n_critiques"] = len(verdicts)
        for v in ("strong", "weak", "unsupported"):
            row[f"crit_{v}"] = verdicts.count(v)
        if r.get("initial_debate") and r.get("debate"):
            before = {(a["criterion"], a["stance"]): a["claim"]
                      for a in r["initial_debate"]["arguments"]}  # fmt: skip
            after = {(a["criterion"], a["stance"]): a["claim"] for a in r["debate"]["arguments"]}
            row["share_args_changed"] = np.mean([after.get(k) != v for k, v in before.items()])
        stats = r.get("llm_stats", {})
        row["prompt_tokens_max"] = max(
            (s.get("prompt_tokens", 0) for s in stats.values()), default=0
        )
        row["output_tokens"] = sum(s.get("output_tokens", 0) for s in stats.values())
        rows.append(row)
    df = sample.merge(pd.DataFrame(rows), on="key", how="inner")
    df["mean_score"] = df[list(CRITERIA)].mean(axis=1)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCORES, index=False)

    judged = df.dropna(subset=list(CRITERIA))
    ok, total = full_text_check(results, sample)
    lines = [
        "# Bloc C : comité complet sur Kaggle — fonctionnement, citations, dispersion (généré)",
        "",
        "Qwen2.5 7B (Ollama), 2 GPU T4 Kaggle, débat Pour / Contre → critique → raffinement",
        "(1 tour) → Juge à grille ancrée. 300 projets de validation + 300 du test, ordre alterné.",
        "**M3 / M4 non calculés ; issue du test non regardée** (en attente de validation).",
        "",
        "## 1. Fonctionnement",
        "",
        f"- Projets notés : {len(judged)} / {len(sample)} (validation "
        f"{(judged.split == 'valid').sum()}, test {(judged.split == 'test').sum()}) ; projets "
        f"avec au moins une erreur consignée : {(df.n_errors > 0).sum()}.",
        f"- Temps par projet (un GPU) : médiane {df.seconds.median():.0f} s, 90e centile "
        f"{df.seconds.quantile(0.9):.0f} s, max {df.seconds.max():.0f} s ; 2 GPU en parallèle ; "
        f"durée totale {prog['elapsed_hours']:.1f} h (quota gratuit Kaggle). Coût : 0 €.",
        f"- Jetons : sortie médiane {df.output_tokens.median():.0f} par projet ; plus long prompt "
        f"{df.prompt_tokens_max.max():.0f} jetons (fenêtre 12 288 : aucune troncature).",
        "",
        "## 2. Citations",
        "",
        f"- Numéros de phrases existants (contrôle sur Kaggle) : "
        f"{df.citation_valid_rate.mean():.1%} des arguments en moyenne.",
        f"- Vérification mot pour mot dans le texte complet masqué (refaite en local) : "
        f"{ok} / {total} = {ok / total:.1%}.",
        "",
        "## 3. Dispersion des notes (0–10), toutes périodes",
        "",
        "| Critère | moyenne | écart-type | min | max | part ≥ 9 | part = 10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in [*CRITERIA, "mean_score"]:
        s = judged[c]
        lines.append(
            f"| {c} | {s.mean():.2f} | {s.std():.2f} | {s.min():.0f} | {s.max():.0f} | "
            f"{(s >= 9).mean():.1%} | {(s == 10).mean():.1%} |"
        )
    corr = judged[list(CRITERIA)].corr().to_numpy()
    off = corr[~np.eye(len(CRITERIA), dtype=bool)]
    lines += [
        "",
        f"Corrélation moyenne entre critères : {off.mean():.2f} (min {off.min():.2f}, "
        f"max {off.max():.2f}). Notes distinctes utilisées : "
        f"{sorted(pd.unique(judged[list(CRITERIA)].to_numpy().ravel()).astype(int).tolist())}.",
        "",
        "## 4. Critique et raffinement",
        "",
        f"- Verdicts du critique : strong {df.crit_strong.sum()}, weak {df.crit_weak.sum()}, "
        f"unsupported {df.crit_unsupported.sum()} (sur {df.n_critiques.sum()}).",
        f"- Part des arguments réécrits au raffinement : {df.share_args_changed.mean():.0%}.",
        "",
    ]
    rt = recognition_table(rec, sample)
    if len(rt):
        known = rt[rt.outcome_known]
        lines += [
            f"## 5. Test de reconnaissance ({len(rt)} projets, réponses comparées en local)",
            "",
            f"- Se dit « reconnu » : {rt.recognized.mean():.0%}.",
            f"- Pays correct : {rt.country_ok.mean():.0%} (le pays est souvent écrit dans le "
            "texte : ce n'est pas une preuve de mémorisation).",
            f"- Année d'approbation à ±1 an : {rt.year_within_1.mean():.0%}.",
            f"- Nom du projet : similarité médiane {rt.name_similarity.median():.2f}, "
            f"≥ 0,8 pour {(rt.name_similarity >= 0.8).sum()} projet(s).",
            f"- Note IEG proposée (≠ « unknown ») : {len(known)} / {len(rt)}."
            + (
                f" Parmi elles, prédiction échec/succès correcte : "
                f"{(known.outcome_guess_fail == known.y_fail.astype(bool)).mean():.0%} "
                f"(taux de succès réel du groupe : {1 - known.y_fail.mean():.0%})."
                if len(known)
                else ""
            ),
            "",
        ]
    va = judged[judged.split == "valid"]
    lines += [
        f"## 6. Lien notes / échec sur la VALIDATION seulement ({len(va)} projets, taux d'échec "
        f"{va.y_fail.mean():.1%}) — diagnostic",
        "",
        "Une note élevée doit aller avec MOINS d'échecs : on mesure l'AUC du score « 10 − note »",
        "pour prédire l'échec (0,5 = hasard).",
        "",
        "| Score | AUC-ROC | AUC-PR (taux de base " + f"{va.y_fail.mean():.3f}) |",
        "| --- | ---: | ---: |",
    ]
    for c in [*CRITERIA, "mean_score"]:
        risk = 10 - va[c]
        lines.append(
            f"| {c} | {roc_auc_score(va.y_fail, risk):.3f} | "
            f"{average_precision_score(va.y_fail, risk):.3f} |"
        )
    lines += [
        "",
        "Pas d'intervalle de confiance ici : diagnostic sur 300 projets. Les chiffres qui",
        "comptent (M3, M4 avec IC) seront réglés sur la validation, puis test une seule fois.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "n_judged": int(len(judged)), "n_sample": int(len(sample)),
        "n_valid": int((judged.split == "valid").sum()),
        "n_test": int((judged.split == "test").sum()),
        "n_projects_with_errors": int((df.n_errors > 0).sum()),
        "median_seconds": float(df.seconds.median()), "elapsed_hours": float(prog["elapsed_hours"]),
        "citation_ids_valid_mean": float(df.citation_valid_rate.mean()),
        "quotes_found": int(ok), "quotes_total": int(total),
        "score_mean": {c: float(judged[c].mean()) for c in CRITERIA},
        "score_std": {c: float(judged[c].std()) for c in CRITERIA},
        "share_ge9": {c: float((judged[c] >= 9).mean()) for c in CRITERIA},
        "mean_corr_between_criteria": float(off.mean()),
        "critic": {"strong": int(df.crit_strong.sum()), "weak": int(df.crit_weak.sum()),
                   "unsupported": int(df.crit_unsupported.sum()),
                   "total": int(df.n_critiques.sum())},
        "share_args_rewritten": float(df.share_args_changed.mean()),
        "valid_auc_roc_by_criterion": {c: float(roc_auc_score(va.y_fail, 10 - va[c]))
                                       for c in CRITERIA},
    }  # fmt: skip
    if len(rt):
        summary["recognition"] = {
            "n": int(len(rt)), "recognized": int(rt.recognized.sum()),
            "ieg_rating_proposed": int(rt.outcome_known.sum()),
            "country_ok": float(rt.country_ok.mean()),
            "name_similarity_ge_08": int((rt.name_similarity >= 0.8).sum()),
        }  # fmt: skip
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
