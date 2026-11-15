"""Expérience de la phase 2, en deux étapes séparées (coffre-fort toujours exclu).

`--stage select` (validation seulement, AUCUNE métrique de test) :
  - M1 : pour chaque ensemble de variables candidat (base, +WDI, +CPIA, +WGI, +tout), réglage
    des hyperparamètres sur la validation ; on garde l'ensemble au meilleur AUC-PR de validation ;
  - M1-texte : réglage de C sur la validation ;
  - écrit experiments/phase2/frozen.json (M1 et M1-texte figés).

`--stage evaluate` (une seule fois, après le gel) :
  - relit frozen.json, réentraîne sur entraînement + validation (≤ 2011), prédit le test
    (2012–2017), calcule métriques, IC bootstrap, écarts appariés, tests de permutation,
    importance des variables et diagnostics de fuite ;
  - écrit experiments/phase2/results.json et reports/phase2_resultats.md.

Les chiffres définitifs de tous les modèles seront calculés une seule fois sur le coffre-fort,
à la fin du projet (décision d'Aya).

Lancer : `uv run python -m investagents.ml.run_phase2 --stage select [--skip-text]`
puis    `uv run python -m investagents.ml.run_phase2 --stage evaluate`
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
FROZEN_PATH = OUT_DIR / "frozen.json"
REPORT_PATH = Path("reports/phase2_resultats.md")
N_BOOT = 2000
CANDIDATE_GROUPS = {
    "base": (),
    "base+wdi": ("wdi",),
    "base+cpia": ("cpia",),
    "base+wgi": ("wgi",),
    "base+wdi+cpia+wgi": ("wdi", "cpia", "wgi"),
}
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


def load_data():
    table = load_modeling_table(with_country_indicators=True)
    table = table[table["split"] != "excluded"].reset_index(drop=True)
    y = target_failure(table)
    split = pd.Series(table["split"].to_numpy(), index=y.index)
    return table, y, split


def load_texts(project_ids) -> pd.Series:
    idx = pd.read_parquet(CORPUS_INDEX_PATH)
    idx = idx[idx["usable"] & idx["project_id"].isin(set(project_ids))]
    texts = {
        r.project_id: read_text(r.doc_id, TEXT_DIR)[: md.TEXT_MAX_CHARS] for r in idx.itertuples()
    }
    return pd.Series(texts, name="text")


# ============================================================== étape 1 : sélection
def select(skip_text: bool) -> None:
    table, y, split = load_data()
    tr, va = (split == "train"), (split == "valid")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8")) if FROZEN_PATH.exists() else {}

    rows = []
    best = None
    for name, groups in CANDIDATE_GROUPS.items():
        x = build_features(table, groups)
        params, log = md.tune(md.make_m1, md.M1_GRID, x[tr], y[tr], x[va], y[va])
        score = float(log["valid_auc_pr"].max())
        log.to_csv(OUT_DIR / f"tuning_m1_{name}.csv", index=False)
        coverage = {
            c: float(x.loc[tr | va, c].notna().mean())
            for c in x.columns
            if c not in build_features(table).columns
        }
        rows.append({"feature_set": name, "valid_auc_pr": score, "params": params})
        print(f"{name:22s} AUC-PR validation {score:.4f}  {params}  couverture {coverage}")
        if best is None or score > best["valid_auc_pr"]:
            best = {
                "feature_set": name,
                "groups": list(groups),
                "params": params,
                "valid_auc_pr": score,
            }
    pd.DataFrame(rows).to_csv(OUT_DIR / "feature_set_selection.csv", index=False)
    frozen["m1"] = best
    print("M1 figé :", best)

    if not skip_text:
        texts = load_texts(y.index[tr | va])
        has = pd.Series(y.index.isin(texts.index), index=y.index)
        tr_t, va_t = tr & has, va & has

        def tx(mask):
            return texts.loc[y.index[mask]].tolist()

        params, log = md.tune(
            md.make_m1_text, md.M1_TEXT_GRID, tx(tr_t), y[tr_t], tx(va_t), y[va_t]
        )
        log.to_csv(OUT_DIR / "tuning_m1_text.csv", index=False)
        frozen["m1_text"] = {
            "params": params,
            "valid_auc_pr": float(log["valid_auc_pr"].max()),
            "n_train": int(tr_t.sum()),
            "n_valid": int(va_t.sum()),
        }
        print("M1-texte figé :", frozen["m1_text"])

    FROZEN_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Écrit : {FROZEN_PATH}")


# ============================================================== étape 2 : évaluation
def evaluate() -> None:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if "m1" not in frozen or "m1_text" not in frozen:
        raise SystemExit("frozen.json incomplet : lancer d'abord --stage select (avec texte).")
    table, y, split = load_data()
    tr, va, te = (split == "train"), (split == "valid"), (split == "test")
    trva = tr | va
    counts = {
        s: {"n": int((split == s).sum()), "failure_rate": float(y[split == s].mean())}
        for s in ["train", "valid", "test"]
    }
    results: dict = {"counts": counts, "frozen": frozen}
    preds = pd.DataFrame({"y": y[te]})

    # M0
    preds["M0"] = md.BaseRate().fit(None, y[trva]).predict_proba(np.zeros(te.sum()))[:, 1]

    # M1 figé
    x = build_features(table, tuple(frozen["m1"]["groups"]))
    p1 = frozen["m1"]["params"]
    m1_tr = md.make_m1(**p1).fit(x[tr], y[tr])
    m1 = md.make_m1(**p1).fit(x[trva], y[trva])
    preds["M1"] = m1.predict_proba(x[te])[:, 1]
    results["m1_diagnostic"] = {
        "train_auc_pr_fit_on_train": compute_metrics(y[tr], m1_tr.predict_proba(x[tr])[:, 1]),
        "valid_auc_pr_fit_on_train": compute_metrics(y[va], m1_tr.predict_proba(x[va])[:, 1]),
    }
    perm = permutation_importance(
        m1, x[te], y[te], scoring="average_precision", n_repeats=20, random_state=md.SEED
    )
    imp = pd.DataFrame(
        {"variable": x.columns, "mean": perm.importances_mean, "std": perm.importances_std}
    ).sort_values("mean", ascending=False)
    imp.to_csv(OUT_DIR / "importance_m1.csv", index=False)

    # M1-texte figé
    texts = load_texts(y.index[trva | te])
    has = pd.Series(y.index.isin(texts.index), index=y.index)
    tr_t, va_t, te_t = tr & has, va & has, te & has

    def tx(mask):
        return texts.loc[y.index[mask]].tolist()

    pt = frozen["m1_text"]["params"]
    mt_tr = md.make_m1_text(**pt).fit(tx(tr_t), y[tr_t])
    mt = md.make_m1_text(**pt).fit(tx(tr_t | va_t), y[tr_t | va_t])
    text_ok = te_t[te].to_numpy()
    preds["M1_text"] = np.nan
    preds.loc[preds.index[text_ok], "M1_text"] = mt.predict_proba(tx(te_t))[:, 1]
    results["m1_text_counts"] = {
        "train": int(tr_t.sum()),
        "valid": int(va_t.sum()),
        "test": int(te_t.sum()),
    }
    results["m1_text_diagnostic"] = {
        "train": compute_metrics(y[tr_t], mt_tr.predict_proba(tx(tr_t))[:, 1]),
        "valid": compute_metrics(y[va_t], mt_tr.predict_proba(tx(va_t))[:, 1]),
    }
    vocab = mt.named_steps["tfidf"].get_feature_names_out()
    words = pd.DataFrame({"mot": vocab, "coef": mt.named_steps["lr"].coef_[0]}).sort_values("coef")
    pd.concat([words.head(30), words.tail(30)]).to_csv(OUT_DIR / "top_words_m1_text.csv")
    flagged = words[words["mot"].isin(SUSPICIOUS_WORDS)]
    preds.to_parquet(OUT_DIR / "test_predictions.parquet")

    # Métriques
    yv = preds["y"].to_numpy()
    results["chance"] = {
        "positive_class": "failure = non satisfaisant (MU/U/HU)",
        "n_test": int(len(yv)),
        "n_positive": int(yv.sum()),
        "base_rate": float(yv.mean()),
        "M1_auc_pr": permutation_pvalue(yv, preds["M1"], "auc_pr", N_BOOT, 6),
        "M1_auc_roc": permutation_pvalue(yv, preds["M1"], "auc_roc", N_BOOT, 7),
    }
    results["test_full"] = {m: bootstrap_ci(yv, preds[m], N_BOOT, 1) for m in ["M0", "M1"]}
    results["diff_full"] = {"M1 - M0": bootstrap_diff(yv, preds["M1"], preds["M0"], N_BOOT, 2)}
    sub = preds[text_ok]
    ys = sub["y"].to_numpy()
    results["chance"]["M1_text_auc_pr"] = permutation_pvalue(
        ys, sub["M1_text"], "auc_pr", N_BOOT, 8
    )
    results["test_text_subset"] = {
        m: bootstrap_ci(ys, sub[m], N_BOOT, 3) for m in ["M0", "M1", "M1_text"]
    }
    results["diff_text_subset"] = {
        "M1_text - M1": bootstrap_diff(ys, sub["M1_text"], sub["M1"], N_BOOT, 4),
        "M1_text - M0": bootstrap_diff(ys, sub["M1_text"], sub["M0"], N_BOOT, 5),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, imp, words, flagged, len(preds), int(text_ok.sum()))


def _table(rows: dict) -> list[str]:
    return [
        "| Modèle | AUC-PR | AUC-ROC | Brier |",
        "| --- | --- | --- | --- |",
        *[
            f"| {m} | {fmt(r['auc_pr'])} | {fmt(r['auc_roc'])} | {fmt(r['brier'])} |"
            for m, r in rows.items()
        ],
    ]


def write_report(results, imp, words, flagged, n_test, n_text) -> None:
    c, fz, ch = results["counts"], results["frozen"], results["chance"]
    lines = [
        "# Phase 2 : résultats (fichier généré automatiquement)",
        "",
        "Généré par `uv run python -m investagents.ml.run_phase2 --stage evaluate`, après gel des",
        "modèles sur la validation. Coffre-fort exclu. Classe positive = projet NON satisfaisant.",
        f"IC 95 % par bootstrap ({N_BOOT} tirages).",
        "",
        "| Split | Exercices | Projets | Taux d'échec |",
        "| --- | --- | ---: | ---: |",
        f"| train | ≤ 2007 | {c['train']['n']} | {c['train']['failure_rate']:.1%} |",
        f"| valid | 2008–2011 | {c['valid']['n']} | {c['valid']['failure_rate']:.1%} |",
        f"| test | 2012–2017 | {c['test']['n']} | {c['test']['failure_rate']:.1%} |",
        "",
        "## Modèles figés (choisis sur la validation)",
        "",
        f"- M1 : variables `{fz['m1']['feature_set']}`, {fz['m1']['params']}, "
        f"AUC-PR validation {fz['m1']['valid_auc_pr']:.3f}",
        f"- M1-texte : {fz['m1_text']['params']}, AUC-PR validation "
        f"{fz['m1_text']['valid_auc_pr']:.3f}",
        "",
        "## Au-dessus du hasard ?",
        "",
        f"- Test : {ch['n_positive']} échecs sur {ch['n_test']} -> taux de base "
        f"{ch['base_rate']:.3f}",
        *[
            f"- Permutation {k} : observé {v['observed']:.3f}, hasard moyen {v['null_mean']:.3f}, "
            f"95e centile {v['null_q95']:.3f}, p = {v['p_value']:.4f}"
            for k, v in ch.items()
            if isinstance(v, dict)
        ],
        "",
        f"## Jeu de test complet (n = {n_test})",
        "",
        *_table(results["test_full"]),
        "",
        *_table(results["diff_full"]),
        "",
        f"## Sous-ensemble du test avec texte (n = {n_text})",
        "",
        *_table(results["test_text_subset"]),
        "",
        *_table(results["diff_text_subset"]),
        "",
        "## Diagnostics de fuite et de surapprentissage",
        "",
        f"- M1 entraîné sur train : AUC-PR train "
        f"{results['m1_diagnostic']['train_auc_pr_fit_on_train']['auc_pr']:.3f}, valid "
        f"{results['m1_diagnostic']['valid_auc_pr_fit_on_train']['auc_pr']:.3f}",
        f"- M1-texte entraîné sur train : AUC-PR train "
        f"{results['m1_text_diagnostic']['train']['auc_pr']:.3f}, valid "
        f"{results['m1_text_diagnostic']['valid']['auc_pr']:.3f}",
        "",
        "Importance par permutation (M1, test, baisse d'AUC-PR) :",
        "",
        imp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Mots qui poussent vers l'échec : " + ", ".join(words.tail(30)["mot"][::-1]),
        "",
        "Mots qui poussent vers le succès : " + ", ".join(words.head(30)["mot"]),
        "",
        "Mots « suspects » (vocabulaire de fin de projet) présents dans le modèle :",
        "",
        flagged.to_markdown(index=False, floatfmt=".3f") if len(flagged) else "aucun",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "evaluate"], required=True)
    parser.add_argument("--skip-text", action="store_true", help="select : M1 seulement")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "select":
        select(args.skip_text)
    else:
        evaluate()


if __name__ == "__main__":
    main()
