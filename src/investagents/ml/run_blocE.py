"""Bloc E : part du pays dans la prédiction (protocole : reports/blocE_pays_protocole.md).

Une seule étape, lancée une seule fois (refuse si experiments/blocE/results.json existe) :
1. contrôle de reproduction : M1-texte d'origine réentraîné (≤ 2007, C figé) doit redonner les
   prédictions brutes d'origine (experiments/mvp/predictions_raw.parquet), sinon arrêt ;
2. M1-texte SANS noms de pays ni adjectifs (liste figée experiments/blocE/country_terms.json) :
   même modèle, même Platt (validation du bloc B), mesure unique sur les 755 projets du test du
   bloc B ; AUC-PR / AUC-ROC avec IC, écart apparié avec M1-texte d'origine ;
3. poids des mots du modèle d'origine (20 vers l'échec, 20 vers la réussite, part « pays » parmi
   les 100 plus lourds) ;
4. effet des 19 facteurs extraits sur la VALIDATION du bloc D (853 projets).

Lancer : `uv run python -m investagents.ml.run_blocE --stage evaluate`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from investagents.agents.masking import country_pattern
from investagents.committee_run.runner import load_jsonl
from investagents.extract.factors import FACTORS
from investagents.ml import models as md
from investagents.ml.calibration import PlattCalibrator
from investagents.ml.metrics import bootstrap_ci, bootstrap_diff
from investagents.ml.run_phase2 import FROZEN_PATH, load_data, load_texts

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/blocE"
RESULTS = OUT / "results.json"
TERMS = OUT / "country_terms.json"
RAW = ROOT / "experiments/mvp/predictions_raw.parquet"
B_TEST = ROOT / "experiments/blocB/test_predictions.parquet"
D_SAMPLE = ROOT / "experiments/blocD/sample.csv"
D_RUNS = ROOT / "experiments/blocD"
M6_RESULTS = ROOT / "experiments/blocD/m6_results.json"
N_BOOT = 2000
SEED_CI, SEED_DIFF, SEED_FACTORS = 41, 42, 43
RISK_HIGH = {"high", "substantial"}


def b_validation_ids() -> pd.Index:
    """Projets de validation du bloc B (ceux de la recalibration de Platt de M1-texte)."""
    from investagents.ml.run_blocB import load as load_b

    df = load_b()
    return df.index[df.split == "valid"]


def text_models(terms: dict) -> dict:
    table, y, split = load_data()
    del table
    tr, va, te = split == "train", split == "valid", split == "test"
    texts = load_texts(y.index[tr | va | te])
    has = pd.Series(y.index.isin(texts.index), index=y.index)
    train_ids = y.index[tr & has]
    params = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))["m1_text"]["params"]
    valid_ids = b_validation_ids()
    b_test = pd.read_parquet(B_TEST).set_index("project_id")
    test_ids = b_test.index

    # 1. contrôle de reproduction
    m0 = md.make_m1_text(**params).fit(texts.loc[train_ids].tolist(), y[train_ids])
    raw = pd.read_parquet(RAW)["m1_text"]
    check = {}
    for name, ids in (("valid", valid_ids), ("test", test_ids)):
        p = m0.predict_proba(texts.loc[ids].tolist())[:, 1]
        check[name] = float(np.max(np.abs(p - raw.loc[ids].to_numpy())))
    if max(check.values()) >= 1e-9:
        raise SystemExit(f"Reproduction impossible : écarts {check}. Arrêt (protocole).")

    # 2. textes sans noms de pays ni adjectifs
    pattern = country_pattern(terms["phrases"])
    n_hits, n_chars, n_removed = [], 0, 0
    clean = {}
    for pid, t in texts.items():
        new, k = pattern.subn(" ", t)
        clean[pid] = new
        if pid in test_ids:
            n_hits.append(k)
            n_chars += len(t)
            n_removed += sum(len(mt.group(0)) for mt in pattern.finditer(t))
    clean = pd.Series(clean)
    m1 = md.make_m1_text(**params).fit(clean.loc[train_ids].tolist(), y[train_ids])
    cal = PlattCalibrator().fit(y[valid_ids],
                                m1.predict_proba(clean.loc[valid_ids].tolist())[:, 1])  # fmt: skip
    p_masked = cal.transform(m1.predict_proba(clean.loc[test_ids].tolist())[:, 1])
    y_test = b_test.y_fail.to_numpy()
    p_orig = b_test.p_fail_m1_text.to_numpy()

    res = {
        "reproduction_max_abs_diff": check,
        "n_train": int(len(train_ids)), "n_valid_platt": int(len(valid_ids)),
        "n_test": int(len(test_ids)), "test_failure_rate": float(y_test.mean()),
        "removed_test": {"mean_expressions_per_doc": float(np.mean(n_hits)),
                         "share_chars_removed": float(n_removed / n_chars)},
        "original": bootstrap_ci(y_test, p_orig, N_BOOT, SEED_CI),
        "no_country": bootstrap_ci(y_test, p_masked, N_BOOT, SEED_CI),
        "diff_no_country_minus_original": bootstrap_diff(y_test, p_masked, p_orig, N_BOOT,
                                                         SEED_DIFF),
    }  # fmt: skip
    a_o, a_m = res["original"]["auc_pr"]["value"], res["no_country"]["auc_pr"]["value"]
    base = res["test_failure_rate"]
    r = (a_o - a_m) / (a_o - base)
    d = res["diff_no_country_minus_original"]["auc_pr"]
    if r >= 0.5:
        verdict = "surtout du pays"
    elif r >= 0.2 or d["ci_high"] < 0:
        verdict = "en partie du pays"
    else:
        verdict = "surtout du contenu du dossier"
    res["share_of_gain_lost"] = float(r)
    res["verdict"] = verdict

    # 3. poids des mots du modèle d'origine
    vocab = m0.named_steps["tfidf"].get_feature_names_out()
    coef = m0.named_steps["lr"].coef_[0]
    country_tok, adj_tok = set(terms["country_tokens"]), set(terms["adjective_tokens"])

    def label(w: str) -> str:
        return "adjectif de pays" if w in adj_tok else ("pays" if w in country_tok else "")

    words = pd.DataFrame({"word": vocab, "coef": coef})
    words["label"] = words.word.map(label)
    fail = words.sort_values("coef", ascending=False).head(20)
    succ = words.sort_values("coef").head(20)
    top100 = words.reindex(words.coef.abs().sort_values(ascending=False).index).head(100)
    res["words"] = {
        "toward_failure": fail.to_dict("records"),
        "toward_success": succ.to_dict("records"),
        "top100_country": int((top100.label == "pays").sum()),
        "top100_adjective": int((top100.label == "adjectif de pays").sum()),
        "top20_failure_country_or_adj": int((fail.label != "").sum()),
        "top20_success_country_or_adj": int((succ.label != "").sum()),
    }
    return res


def present(f, a: dict, median: float | None) -> bool:
    ok = a["status"] == "found" and a["consistent"] and a.get("ids_valid", False)
    if not ok:
        return False
    if f.kind == "yes_no":
        return a["value"] == "yes"
    if f.kind == "list":
        return a["value"] is not None and int(a["value"]) > median
    if f.name == "env_category":
        return a["value"] == "A"
    return a["value"] in RISK_HIGH


def factor_effects() -> tuple[list[dict], int]:
    from investagents.ml.run_m6 import load as load_m6

    valid = load_m6().query("split == 'valid'")[["key", "y_fail"]]
    recs = {}
    for path in sorted(D_RUNS.glob("kaggle_*/results.jsonl")):  # même lecture que run_m6
        for r in load_jsonl(path):
            if r.get("factors"):
                recs[r["key"]] = r["factors"]
    weights = json.loads(M6_RESULTS.read_text(encoding="utf-8"))["weights"]["m5"]
    y = valid.y_fail.to_numpy()
    rng = np.random.default_rng(SEED_FACTORS)
    boot = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    out = []
    for f in FACTORS:
        answers = [recs[k][f.name] for k in valid.key]
        median = None
        if f.kind == "list":
            found = [int(a["value"]) for a in answers if a["status"] == "found"
                     and a["consistent"] and a.get("ids_valid", False) and a["value"] is not None]
            median = float(np.median(found)) if found else 0.0  # fmt: skip
        x = np.array([present(f, a, median) for a in answers])
        if x.sum() == 0 or (~x).sum() == 0:
            continue
        eff = y[x].mean() - y[~x].mean()
        samples = []
        for idx in boot:
            xb, yb = x[idx], y[idx]
            if xb.any() and (~xb).any():
                samples.append(yb[xb].mean() - yb[~xb].mean())
        lo, hi = np.quantile(samples, [0.025, 0.975])
        w = weights.get(f.name)
        if f.name == "env_category":
            w = weights.get("env_A")
        out.append({"factor": f.name, "kind": f.kind, "n_present": int(x.sum()),
                    "fail_present": float(y[x].mean()), "fail_absent": float(y[~x].mean()),
                    "effect": float(eff), "ci_low": float(lo), "ci_high": float(hi),
                    "median_threshold": median, "m5_weight": w})  # fmt: skip
    return sorted(out, key=lambda r: -r["effect"]), int(len(valid))


def stage_evaluate() -> None:
    if RESULTS.exists():
        raise SystemExit("Déjà calculé : le test ne se lit qu'une fois (protocole du bloc E).")
    terms = json.loads(TERMS.read_text(encoding="utf-8"))
    res = {"terms_sha256": terms["sha256"], **text_models(terms)}
    res["factors_valid"], res["n_valid_factors"] = factor_effects()
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("verdict", "share_of_gain_lost",
                                          "reproduction_max_abs_diff")}, indent=1))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["evaluate"], required=True)
    parser.parse_args()
    stage_evaluate()


if __name__ == "__main__":
    main()
