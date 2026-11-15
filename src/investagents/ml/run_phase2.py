"""Expérience de la phase 2 : M0, M1, M1-texte sur le split temporel, coffre-fort exclu.

Étapes :
1. table de modélisation (coffre-fort retiré dès le chargement) et variables ex ante ;
2. réglage sur la validation (2008–2011), réentraînement sur ≤ 2011, prédiction sur 2012–2017 ;
3. métriques + IC 95 % bootstrap, écarts appariés, importance des variables ;
4. diagnostics de fuite : scores entraînement / validation / test, mots les plus influents.

Sorties : experiments/phase2/*.csv|json et reports/phase2_resultats.md (généré).
Lancer : `uv run python -m investagents.ml.run_phase2 [--skip-text]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from investagents.data.corpus import CORPUS_INDEX_PATH, TEXT_DIR, read_text
from investagents.ml import models as md
from investagents.ml.features import build_features, target_failure
from investagents.ml.metrics import (
    bootstrap_ci,
    bootstrap_diff,
    compute_metrics,
    permutation_pvalue,
)
from investagents.ml.split import load_modeling_table

OUT_DIR = Path("experiments/phase2")
REPORT_PATH = Path("reports/phase2_resultats.md")
N_BOOT = 2000
SUSPICIOUS_WORDS = {
    "completion",
    "icr",
    "icrr",
    "ppar",
    "restructuring",
    "restructured",
    "cancelled",
    "cancellation",
    "unsatisfactory",
    "rating",
    "rated",
    "outcome",
    "evaluation",
    "closing",
    "extension",
}


def fmt(m: dict) -> str:
    return f"{m['value']:.3f} [{m['ci_low']:.3f} ; {m['ci_high']:.3f}]"


def load_texts(project_ids) -> pd.Series:
    idx = pd.read_parquet(CORPUS_INDEX_PATH)
    idx = idx[idx["usable"] & idx["project_id"].isin(set(project_ids))]
    texts = {
        r.project_id: read_text(r.doc_id, TEXT_DIR)[: md.TEXT_MAX_CHARS] for r in idx.itertuples()
    }
    return pd.Series(texts, name="text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-text", action="store_true", help="ne pas lancer M1-texte")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = load_modeling_table()
    table = table[table["split"] != "excluded"].reset_index(drop=True)
    x_all = build_features(table)
    y_all = target_failure(table)
    split = pd.Series(table["split"].to_numpy(), index=x_all.index)
    tr, va, te = (split == "train"), (split == "valid"), (split == "test")
    trva = tr | va
    counts = {
        s: {"n": int((split == s).sum()), "failure_rate": float(y_all[split == s].mean())}
        for s in ["train", "valid", "test"]
    }
    print("Effectifs :", counts)

    results: dict = {"counts": counts}
    preds = pd.DataFrame({"y": y_all[te]})

    # ---------------- M0
    m0 = md.BaseRate().fit(None, y_all[trva])
    preds["M0"] = m0.predict_proba(x_all[te])[:, 1]

    # ---------------- M1
    best_m1, log_m1 = md.tune(md.make_m1, md.M1_GRID, x_all[tr], y_all[tr], x_all[va], y_all[va])
    log_m1.to_csv(OUT_DIR / "tuning_m1.csv", index=False)
    print("M1 meilleurs hyperparamètres :", best_m1)
    m1_tr = md.make_m1(**best_m1).fit(x_all[tr], y_all[tr])
    m1 = md.make_m1(**best_m1).fit(x_all[trva], y_all[trva])
    preds["M1"] = m1.predict_proba(x_all[te])[:, 1]
    results["m1_params"] = best_m1
    results["m1_diagnostic"] = {
        "train_fit_on_train": compute_metrics(y_all[tr], m1_tr.predict_proba(x_all[tr])[:, 1]),
        "valid_fit_on_train": compute_metrics(y_all[va], m1_tr.predict_proba(x_all[va])[:, 1]),
        "test_fit_on_train_valid": compute_metrics(preds["y"], preds["M1"]),
    }

    perm = permutation_importance(
        m1,
        x_all[te],
        y_all[te],
        scoring="average_precision",
        n_repeats=20,
        random_state=md.SEED,
    )
    imp_m1 = pd.DataFrame(
        {"variable": x_all.columns, "mean": perm.importances_mean, "std": perm.importances_std}
    ).sort_values("mean", ascending=False)
    imp_m1.to_csv(OUT_DIR / "importance_m1.csv", index=False)

    # ---------------- M1-texte
    text_ok = None
    if not args.skip_text:
        texts = load_texts(x_all.index[trva | te])
        has_text = pd.Series(x_all.index.isin(texts.index), index=x_all.index)
        tr_t, va_t, te_t = tr & has_text, va & has_text, te & has_text
        print(f"Projets avec texte : train {tr_t.sum()}, valid {va_t.sum()}, test {te_t.sum()}")

        def tx(mask):
            return texts.loc[x_all.index[mask]].tolist()

        best_t, log_t = md.tune(
            md.make_m1_text, md.M1_TEXT_GRID, tx(tr_t), y_all[tr_t], tx(va_t), y_all[va_t]
        )
        log_t.to_csv(OUT_DIR / "tuning_m1_text.csv", index=False)
        print("M1-texte meilleur C :", best_t)
        mt_tr = md.make_m1_text(**best_t).fit(tx(tr_t), y_all[tr_t])
        mt = md.make_m1_text(**best_t).fit(tx(tr_t | va_t), y_all[tr_t | va_t])
        text_ok = te_t[te].to_numpy()
        preds["M1_text"] = np.nan
        preds.loc[preds.index[text_ok], "M1_text"] = mt.predict_proba(tx(te_t))[:, 1]
        results["m1_text_params"] = best_t
        results["m1_text_counts"] = {
            "train": int(tr_t.sum()),
            "valid": int(va_t.sum()),
            "test": int(te_t.sum()),
        }
        results["m1_text_diagnostic"] = {
            "train_fit_on_train": compute_metrics(
                y_all[tr_t], mt_tr.predict_proba(tx(tr_t))[:, 1]
            ),
            "valid_fit_on_train": compute_metrics(
                y_all[va_t], mt_tr.predict_proba(tx(va_t))[:, 1]
            ),
        }
        vocab = mt.named_steps["tfidf"].get_feature_names_out()
        coef = mt.named_steps["lr"].coef_[0]
        words = pd.DataFrame({"mot": vocab, "coef": coef}).sort_values("coef")
        top = pd.concat([words.head(30), words.tail(30)])
        top.to_csv(OUT_DIR / "top_words_m1_text.csv", index=False)
        flagged = words[words["mot"].isin(SUSPICIOUS_WORDS)]
        flagged.to_csv(OUT_DIR / "suspicious_words_m1_text.csv", index=False)

    preds.to_parquet(OUT_DIR / "test_predictions.parquet")

    # ---------------- métriques et IC
    y = preds["y"].to_numpy()
    results["test_full"] = {
        m: bootstrap_ci(y, preds[m], n_boot=N_BOOT, seed=1) for m in ["M0", "M1"]
    }
    results["chance"] = {
        "positive_class": "failure = non satisfaisant (MU/U/HU)",
        "n_test": int(len(y)),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()),
        "M1_auc_pr_permutation": permutation_pvalue(y, preds["M1"], "auc_pr", N_BOOT, 6),
        "M1_auc_roc_permutation": permutation_pvalue(y, preds["M1"], "auc_roc", N_BOOT, 7),
    }
    results["diff_full"] = {"M1 - M0": bootstrap_diff(y, preds["M1"], preds["M0"], N_BOOT, 2)}
    if text_ok is not None:
        sub = preds[text_ok]
        ys = sub["y"].to_numpy()
        results["test_text_subset"] = {
            m: bootstrap_ci(ys, sub[m], n_boot=N_BOOT, seed=3) for m in ["M0", "M1", "M1_text"]
        }
        results["diff_text_subset"] = {
            "M1_text - M1": bootstrap_diff(ys, sub["M1_text"], sub["M1"], N_BOOT, 4),
            "M1_text - M0": bootstrap_diff(ys, sub["M1_text"], sub["M0"], N_BOOT, 5),
        }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---------------- rapport généré
    lines = [
        "# Phase 2 : résultats bruts (fichier généré automatiquement)",
        "",
        "Généré par `uv run python -m investagents.ml.run_phase2`. Coffre-fort exclu.",
        "Classe positive = projet NON satisfaisant. IC 95 % par bootstrap "
        f"({N_BOOT} tirages).",
        "",
        "| Split | Exercices d'approbation | Projets | Taux d'échec |",
        "| --- | --- | ---: | ---: |",
        f"| train | ≤ 2007 | {counts['train']['n']} | {counts['train']['failure_rate']:.1%} |",
        f"| valid | 2008–2011 | {counts['valid']['n']} | {counts['valid']['failure_rate']:.1%} |",
        f"| test | 2012–2017 | {counts['test']['n']} | {counts['test']['failure_rate']:.1%} |",
        "",
        "## Au-dessus du hasard ?",
        "",
        f"- Classe positive : {results['chance']['positive_class']}",
        f"- Test : {results['chance']['n_positive']} positifs sur {results['chance']['n_test']}"
        f" -> taux de base {results['chance']['base_rate']:.3f}"
        " (= AUC-PR d'un classement au hasard)",
        *[
            f"- Test de permutation M1 {k.split('_')[1].upper()}-{k.split('_')[2].upper()} : "
            f"observé {v['observed']:.3f}, hasard moyen {v['null_mean']:.3f}, "
            f"95e centile du hasard {v['null_q95']:.3f}, p = {v['p_value']:.4f}"
            for k, v in results["chance"].items()
            if k.endswith("permutation")
        ],
        "",
        f"## Jeu de test complet (n = {len(preds)})",
        "",
        "| Modèle | AUC-PR | AUC-ROC | Brier |",
        "| --- | --- | --- | --- |",
        *[
            f"| {m} | {fmt(r['auc_pr'])} | {fmt(r['auc_roc'])} | {fmt(r['brier'])} |"
            for m, r in results["test_full"].items()
        ],
        "",
        "| Écart apparié | AUC-PR | AUC-ROC | Brier |",
        "| --- | --- | --- | --- |",
        *[
            f"| {k} | {fmt(r['auc_pr'])} | {fmt(r['auc_roc'])} | {fmt(r['brier'])} |"
            for k, r in results["diff_full"].items()
        ],
        "",
    ]
    if text_ok is not None:
        lines += [
            f"## Sous-ensemble du test avec texte (n = {int(text_ok.sum())})",
            "",
            "| Modèle | AUC-PR | AUC-ROC | Brier |",
            "| --- | --- | --- | --- |",
            *[
                f"| {m} | {fmt(r['auc_pr'])} | {fmt(r['auc_roc'])} | {fmt(r['brier'])} |"
                for m, r in results["test_text_subset"].items()
            ],
            "",
            "| Écart apparié | AUC-PR | AUC-ROC | Brier |",
            "| --- | --- | --- | --- |",
            *[
                f"| {k} | {fmt(r['auc_pr'])} | {fmt(r['auc_roc'])} | {fmt(r['brier'])} |"
                for k, r in results["diff_text_subset"].items()
            ],
            "",
        ]
    lines += [
        "## Diagnostics",
        "",
        f"- M1 hyperparamètres : {best_m1}",
        "- M1 (entraîné sur train) : AUC-PR train "
        f"{results['m1_diagnostic']['train_fit_on_train']['auc_pr']:.3f}, valid "
        f"{results['m1_diagnostic']['valid_fit_on_train']['auc_pr']:.3f}",
        "",
        "Importance par permutation (M1, test, baisse d'AUC-PR) :",
        "",
        imp_m1.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    if text_ok is not None:
        d = results["m1_text_diagnostic"]
        lines += [
            f"- M1-texte C = {results['m1_text_params']['C']} ; AUC-PR train "
            f"{d['train_fit_on_train']['auc_pr']:.3f}, "
            f"valid {d['valid_fit_on_train']['auc_pr']:.3f}",
            "",
            "Mots qui poussent vers l'échec (coefficients les plus positifs) :",
            "",
            ", ".join(words.tail(30)["mot"][::-1]),
            "",
            "Mots qui poussent vers le succès (coefficients les plus négatifs) :",
            "",
            ", ".join(words.head(30)["mot"]),
            "",
            "Mots « suspects » (vocabulaire post-projet) présents dans le modèle :",
            "",
            flagged.to_markdown(index=False, floatfmt=".3f") if len(flagged) else "aucun",
            "",
        ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
