"""M3 et M4 : les notes du comité apportent-elles une information ? (bloc C, expérience centrale)

Données : les 600 projets notés par le comité (300 validation 2008–2011, 300 test 2012–2017).
- M0 : taux de base (constant) — le plancher.
- Texte : modèle de texte figé au bloc B (M1-texte, entraîné sur ≤ 2007), probabilités
  recalibrées (Platt) sur les 300 projets de validation.
- M3 : régression logistique sur les 5 notes du Juge (standardisées).
- M4 : régression logistique sur [logit du modèle de texte, 5 notes]. Écart assumé au cahier
  des charges (gradient boosting hybride) : avec 300 projets d'apprentissage, un gradient boosting
  sur-apprend ; un empilement logistique a 6 paramètres.
Réglage (C de la régression) : validation croisée à 5 plis DANS les 300 projets de validation,
critère AUC-PR ; puis réapprentissage sur les 300. Le test n'intervient dans aucun choix.

Deux étapes :
- `--stage select` : réglage sur la validation, écrit experiments/blocC/models_frozen.json ;
- `--stage evaluate` : évaluation UNIQUE sur les 300 du test (refuse de tourner si le fichier
  figé n'existe pas), IC à 95 % par bootstrap, écarts appariés M4 − texte et M3 − M0.

Lancer : `uv run python -m investagents.ml.run_blocC_models --stage select|evaluate`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from investagents.ml.calibration import PlattCalibrator, ece
from investagents.ml.metrics import bootstrap_ci, bootstrap_diff, compute_metrics
from investagents.schemas.committee import CRITERIA

SCORES = Path("experiments/blocC/scores.parquet")
RAW_PRED = Path("experiments/mvp/predictions_raw.parquet")
OUT = Path("experiments/blocC")
FROZEN = OUT / "models_frozen.json"
REPORT = Path("reports/blocC_m3_m4.md")
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
N_BOOT = 2000
SEED = 21


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load() -> pd.DataFrame:
    df = pd.read_parquet(SCORES).dropna(subset=list(CRITERIA))
    raw = pd.read_parquet(RAW_PRED)[["m1_text"]]
    df = df.merge(raw, left_on="project_id", right_index=True, how="left")
    assert df.m1_text.notna().all(), "projet sans prédiction du modèle de texte"
    return df


def features(df: pd.DataFrame, name: str) -> np.ndarray:
    notes = df[list(CRITERIA)].to_numpy(dtype=float)
    if name == "m3":
        return notes
    if name == "m4":
        return np.column_stack([logit(df.m1_text), notes])
    raise ValueError(name)


def make(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def cv_auc_pr(x, y, c) -> float:
    from sklearn.metrics import average_precision_score

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(make(c), x, y, cv=cv, method="predict_proba")[:, 1]
    return float(average_precision_score(y, p))


def stage_select() -> None:
    df = load()
    va = df[df.split == "valid"]
    y = va.y_fail.to_numpy()
    out = {"n_valid": len(va), "valid_failure_rate": float(y.mean()), "models": {}}
    for name in ("m3", "m4"):
        x = features(va, name)
        scores = {c: cv_auc_pr(x, y, c) for c in C_GRID}
        best = max(scores, key=scores.get)
        out["models"][name] = {"C": best, "cv_auc_pr": scores}
    out["text_cv_reference"] = {
        "valid_auc_pr_text_alone": compute_metrics(y, va.m1_text)["auc_pr"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def stage_evaluate() -> None:
    if not FROZEN.exists():
        raise SystemExit("Lancer d'abord --stage select (réglage sur la validation).")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    df = load()
    va, te = df[df.split == "valid"], df[df.split == "test"]
    yv, yt = va.y_fail.to_numpy(), te.y_fail.to_numpy()

    preds, weights = {}, {}
    preds["m0"] = np.full(len(te), yv.mean())
    cal = PlattCalibrator().fit(yv, va.m1_text.to_numpy())
    preds["texte"] = cal.transform(te.m1_text.to_numpy())
    for name in ("m3", "m4"):
        model = make(frozen["models"][name]["C"]).fit(features(va, name), yv)
        preds[name] = model.predict_proba(features(te, name))[:, 1]
        lr = model[-1]
        cols = list(CRITERIA) if name == "m3" else ["logit_texte", *CRITERIA]
        weights[name] = dict(zip(cols, lr.coef_[0].round(4).tolist(), strict=True))
        weights[name]["intercept"] = round(float(lr.intercept_[0]), 4)

    res = {"n_test": len(te), "test_failure_rate": float(yt.mean()), "models": {}, "diffs": {}}
    for name, p in preds.items():
        ci = bootstrap_ci(yt, p, N_BOOT, seed=SEED) if name != "m0" else None
        res["models"][name] = {
            "metrics": compute_metrics(yt, p) if name != "m0" else None,
            "ci": ci,
            "ece": ece(yt, p),
            "brier": float(np.mean((p - yt) ** 2)),
        }
    res["diffs"]["m4 - texte"] = bootstrap_diff(yt, preds["m4"], preds["texte"], N_BOOT, SEED)
    res["diffs"]["m3 - m0"] = bootstrap_diff(yt, preds["m3"], preds["m0"], N_BOOT, SEED)
    res["diffs"]["m3 - texte"] = bootstrap_diff(yt, preds["m3"], preds["texte"], N_BOOT, SEED)
    res["weights"] = weights
    (OUT / "models_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    test_preds = {"key": te.key, "y_fail": yt, **{f"p_{k}": v for k, v in preds.items()}}
    pd.DataFrame(test_preds).to_parquet(OUT / "models_test_predictions.parquet", index=False)
    write_report(frozen, res)


def fmt(d) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f} ; {d['ci_high']:.3f}]"


def write_report(frozen, res) -> None:
    base = res["test_failure_rate"]
    lines = [
        "# Bloc C : M3 et M4 — apport des notes du comité (généré)",
        "",
        f"Réglage : validation croisée à 5 plis dans les {frozen['n_valid']} projets de validation",
        f"(taux d'échec {frozen['valid_failure_rate']:.1%}). Test : {res['n_test']} projets, taux",
        f"d'échec {base:.1%}, calculé une seule fois. Classe positive = échec.",
        "",
        "| Modèle | C | AUC-PR validation (VC) |",
        "| --- | ---: | ---: |",
    ]
    for name, m in frozen["models"].items():
        lines.append(f"| {name} | {m['C']} | {m['cv_auc_pr'][str(m['C'])]:.3f} |"
                     if str(m["C"]) in m["cv_auc_pr"] else
                     f"| {name} | {m['C']} | {max(m['cv_auc_pr'].values()):.3f} |")  # fmt: skip
    lines += [
        "",
        "## Test (une seule fois)",
        "",
        "| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |",
        "| --- | --- | --- | ---: | ---: |",
        f"| M0 (taux de base) | {base:.3f} (= taux d'échec) | 0.500 | "
        f"{res['models']['m0']['brier']:.4f} | {res['models']['m0']['ece']:.3f} |",
    ]
    labels = {"texte": "Texte (M1-texte figé, Platt)", "m3": "M3 (5 notes)",
              "m4": "M4 (texte + 5 notes)"}  # fmt: skip
    for name in ("texte", "m3", "m4"):
        m = res["models"][name]
        lines.append(
            f"| {labels[name]} | {fmt(m['ci']['auc_pr'])} | {fmt(m['ci']['auc_roc'])} | "
            f"{m['brier']:.4f} | {m['ece']:.3f} |"
        )
    lines += [
        "",
        "| Écart apparié (même tirages) | AUC-PR | AUC-ROC | Brier |",
        "| --- | --- | --- | --- |",
        *[
            f"| {k} | {fmt(d['auc_pr'])} | {fmt(d['auc_roc'])} | {fmt(d['brier'])} |"
            for k, d in res["diffs"].items()
        ],
        "",
        "## Poids appris (coefficients sur variables standardisées ; positif = plus d'échec)",
        "",
    ]
    for name, w in res["weights"].items():
        lines.append(f"- {name} : " + ", ".join(f"{k} {v:+.3f}" for k, v in w.items()))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "evaluate"], required=True)
    stage = parser.parse_args().stage
    stage_select() if stage == "select" else stage_evaluate()


if __name__ == "__main__":
    main()
