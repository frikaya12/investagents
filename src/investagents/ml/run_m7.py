"""M7 : apport prédictif des SORTIES INTERMÉDIAIRES du comité (bloc C+, point 2).

Aucun nouvel appel LLM : variables tirées de experiments/blocC/kaggle_run1/results.jsonl.
- Par critère et par position (Pour / Contre) : l'argument initial est-il jugé « strong » par le
  critique ? (10 variables 0/1 : le débat produit 1 argument Pour et 1 Contre par critère)
- Part des arguments jugés « unsupported » ; part des arguments réécrits au raffinement.
- TF-IDF du texte des arguments Contre (version finale, après raffinement).
M7 = régression logistique sur [logit du modèle de texte M1-texte, variables ci-dessus, TF-IDF].
Réglage : validation croisée à 5 plis DANS les 300 projets de validation (critère AUC-PR) sur
C et sur le poids du bloc TF-IDF (0 = bloc exclu). Puis test UNE fois sur les 300 du test.
C'est le 2e usage de ces 300 projets de test (le 1er : M3 / M4) ; écrit dans le rapport.

`--stage select`   : validation seulement → experiments/blocC/m7_frozen.json
`--stage evaluate` : test une fois (refuse si déjà fait) → m7_results.json, reports/blocC_m7.md

Lancer : `uv run python -m investagents.ml.run_m7 --stage select|evaluate`
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from investagents.committee_run.runner import load_jsonl
from investagents.ml.calibration import PlattCalibrator, ece
from investagents.ml.metrics import bootstrap_ci, bootstrap_diff, compute_metrics
from investagents.ml.run_blocC_models import load as load_scores
from investagents.ml.run_blocC_models import logit
from investagents.schemas.committee import CRITERIA

RESULTS = Path("experiments/blocC/kaggle_run1/results.jsonl")
OUT = Path("experiments/blocC")
FROZEN = OUT / "m7_frozen.json"
RESULTS_OUT = OUT / "m7_results.json"
REPORT = Path("reports/blocC_m7.md")
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
TFIDF_WEIGHTS = [0.0, 0.3, 1.0]
N_BOOT = 2000
SEED = 21
NUM_COLS = [
    *[f"strong_{s}_{c}" for c in CRITERIA for s in ("pro", "contra")],
    "share_unsupported",
    "share_rewritten",
]


def critique_features(r: dict) -> dict:
    """Variables d'un projet à partir de sa sortie du comité (aucun chiffre du LLM : comptes)."""
    initial = r["initial_debate"]["arguments"]
    row = {f"strong_{s}_{c}": 0 for c in CRITERIA for s in ("pro", "contra")}
    verdicts = []
    for h in r.get("rounds_history", [])[:1]:  # 1er tour : critiques des arguments initiaux
        args = h.get("arguments") or initial
        for cr in h["critiques"]:
            a = args[cr["argument_index"]]
            verdicts.append(cr["verdict"])
            if cr["verdict"] == "strong":
                row[f"strong_{a['stance']}_{a['criterion']}"] = 1
    row["share_unsupported"] = verdicts.count("unsupported") / max(len(verdicts), 1)
    before = {(a["criterion"], a["stance"]): a["claim"] for a in initial}
    after = {(a["criterion"], a["stance"]): a["claim"] for a in r["debate"]["arguments"]}
    row["share_rewritten"] = float(np.mean([after.get(k) != v for k, v in before.items()]))
    row["contra_text"] = " ".join(
        a["claim"] for a in r["debate"]["arguments"] if a["stance"] == "contra"
    )
    return row


def load() -> pd.DataFrame:
    df = load_scores()  # 600 projets notés + m1_text brut + y_fail + split
    rs = {r["key"]: r for r in load_jsonl(RESULTS)}
    feats = pd.DataFrame([{"key": k, **critique_features(r)} for k, r in rs.items()])
    df = df.merge(feats, on="key", how="left")
    assert df.contra_text.notna().all()
    df["logit_text"] = logit(df.m1_text)
    return df


def make(c: float, tfidf_weight: float):
    blocks = [("num", StandardScaler(), ["logit_text", *NUM_COLS])]
    weights = {"num": 1.0}
    if tfidf_weight > 0:
        tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True,
                                max_features=3000, stop_words="english")  # fmt: skip
        blocks.append(("txt", tfidf, "contra_text"))
        weights["txt"] = tfidf_weight
    ct = ColumnTransformer(blocks, transformer_weights=weights)
    return make_pipeline(ct, LogisticRegression(C=c, max_iter=5000))


def cv_auc_pr(x: pd.DataFrame, y, c, w) -> float:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(make(c, w), x, y, cv=cv, method="predict_proba")[:, 1]
    return float(average_precision_score(y, p))


def stage_select() -> None:
    df = load()
    va = df[df.split == "valid"]
    y = va.y_fail.to_numpy()
    grid = {f"C={c}|tfidf={w}": cv_auc_pr(va, y, c, w)
            for c, w in itertools.product(C_GRID, TFIDF_WEIGHTS)}  # fmt: skip
    best = max(grid, key=grid.get)
    c, w = (float(p.split("=")[1]) for p in best.split("|"))
    # référence : texte seul, même validation croisée (logit_text seul, aucune variable agent)
    ref = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), va[["logit_text"]],
        y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba",
    )[:, 1]  # fmt: skip
    out = {
        "n_valid": len(va),
        "C": c,
        "tfidf_weight": w,
        "cv_auc_pr": grid[best],
        "cv_auc_pr_text_alone": float(average_precision_score(y, ref)),
        "grid": grid,
    }
    FROZEN.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=2))


def stage_evaluate() -> None:
    if not FROZEN.exists():
        raise SystemExit("Lancer d'abord --stage select.")
    if RESULTS_OUT.exists():
        raise SystemExit(f"Déjà évalué ({RESULTS_OUT}) : le test ne se calcule qu'une fois.")
    fr = json.loads(FROZEN.read_text(encoding="utf-8"))
    df = load()
    va, te = df[df.split == "valid"], df[df.split == "test"]
    yv, yt = va.y_fail.to_numpy(), te.y_fail.to_numpy()
    p_text = PlattCalibrator().fit(yv, va.m1_text.to_numpy()).transform(te.m1_text.to_numpy())
    model = make(fr["C"], fr["tfidf_weight"]).fit(va, yv)
    p_m7 = model.predict_proba(te)[:, 1]
    coefs = model[-1].coef_[0][: 1 + len(NUM_COLS)]
    res = {
        "n_test": len(te),
        "test_failure_rate": float(yt.mean()),
        "models": {
            name: {"metrics": compute_metrics(yt, p), "ci": bootstrap_ci(yt, p, N_BOOT, seed=SEED),
                   "brier": float(np.mean((p - yt) ** 2)), "ece": ece(yt, p)}  # fmt: skip
            for name, p in (("texte", p_text), ("m7", p_m7))
        },
        "diff_m7_texte": bootstrap_diff(yt, p_m7, p_text, N_BOOT, SEED),
        "coefs_num": dict(zip(["logit_text", *NUM_COLS], coefs.round(4).tolist(), strict=True)),
    }
    RESULTS_OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(fr, res)


def fmt(d) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f} ; {d['ci_high']:.3f}]"


def write_report(fr: dict, res: dict) -> None:
    lines = [
        "# Bloc C+ : M7 — sorties intermédiaires du comité (généré)",
        "",
        "Variables : arguments jugés « strong » par critère et position, part « unsupported »,",
        "part réécrite au raffinement, TF-IDF des arguments Contre ; plus le logit de M1-texte.",
        f"Réglage par validation croisée à 5 plis dans les {fr['n_valid']} projets de validation :",
        f"C = {fr['C']}, poids du TF-IDF = {fr['tfidf_weight']} ; AUC-PR (VC) {fr['cv_auc_pr']:.3f}"
        f" contre {fr['cv_auc_pr_text_alone']:.3f} pour le texte seul (même VC).",
        "",
        f"**Test : {res['n_test']} projets (taux d'échec {res['test_failure_rate']:.1%}), calculé "
        "une seule fois. C'est le 2e usage de ces 300 projets de test (après M3 / M4).**",
        "",
        "| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    labels = {"texte": "Texte (M1-texte, Platt)", "m7": "M7 (texte + sorties du comité)"}
    for k, m in res["models"].items():
        lines.append(f"| {labels[k]} | {fmt(m['ci']['auc_pr'])} | {fmt(m['ci']['auc_roc'])} | "
                     f"{m['brier']:.4f} | {m['ece']:.3f} |")  # fmt: skip
    d = res["diff_m7_texte"]
    lines += [
        "",
        "| Écart apparié | AUC-PR | AUC-ROC | Brier |",
        "| --- | --- | --- | --- |",
        f"| M7 − texte | {fmt(d['auc_pr'])} | {fmt(d['auc_roc'])} | {fmt(d['brier'])} |",
        "",
        "Coefficients (variables standardisées, positif = plus d'échec) : "
        + ", ".join(f"{k} {v:+.3f}" for k, v in res["coefs_num"].items()),
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "evaluate"], required=True)
    stage_select() if parser.parse_args().stage == "select" else stage_evaluate()


if __name__ == "__main__":
    main()
