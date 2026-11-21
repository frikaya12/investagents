"""Bloc B : variantes de modèles de texte, choisies sur la VALIDATION, puis test une seule fois.

Tous les modèles sont entraînés sur l'entraînement (≤ 2007), comparés sur la validation
(2008–2011) par AUC-PR (classe « échec »), sur les MÊMES projets (ceux qui ont un texte).
Candidats :
- `m1_text_full`      : M1-texte de la phase 2 (TF-IDF, 150 000 premiers caractères), référence ;
- `tfidf_sections`    : TF-IDF sur les sections clés (8 000 caractères) ;
- `tfidf_masked`      : idem, pays / lieux / nationalités masqués (mesure le signal hors pays) ;
- `emb_sections`      : embeddings de phrases all-MiniLM-L6-v2 (moyenne des morceaux) + régression
                        logistique ;
- `stack`             : régression logistique sur [logit p(M1), logit p(meilleur texte)] ; son
                        score de validation est estimé par validation croisée à 5 plis À
                        L'INTÉRIEUR de la validation (sinon évalué sur ses propres données).
Choix : meilleur AUC-PR de validation ; écrit experiments/blocB/frozen.json.

`--stage evaluate` (une seule fois) : recalibration de Platt sur la validation ; sur le test :
AUC-PR, AUC-ROC, Brier, ECE, IC 95 %, écarts appariés contre M1 et M1-texte (versions entraînées
sur ≤ 2007, mêmes projets) ; la variante masquée est aussi évaluée, à titre de diagnostic ;
backtest de portefeuille avec l'objectif figé du bloc A.

Lancer : `uv run python -m investagents.ml.run_blocB --stage select` puis `--stage evaluate`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from investagents.ml import models as md
from investagents.ml.calibration import PlattCalibrator, ece
from investagents.ml.metrics import bootstrap_ci, bootstrap_diff, compute_metrics

OUT_DIR = Path("experiments/blocB")
FROZEN = OUT_DIR / "frozen.json"
REPORT = Path("reports/blocB_resultats.md")
SECTIONS = Path("data/processed/sections_corpus.parquet")
EMB_CACHE = Path("data/processed/emb_sections_v2.npy")  # v2 : sans troncature
RAW_PRED = Path("experiments/mvp/predictions_raw.parquet")
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
EMB_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 800
N_BOOT = 2000


# ------------------------------------------------------------------ données
def load() -> pd.DataFrame:
    """Une ligne par projet avec texte : split, y_fail, prédictions M1 / M1-texte, sections."""
    raw = pd.read_parquet(RAW_PRED)  # index = project_id ; modèles entraînés sur ≤ 2007
    sec = pd.read_parquet(SECTIONS).set_index("project_id")
    df = raw.join(sec, how="inner")
    df = df[df.split.isin(["train", "valid", "test"])]
    return df


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def split_to_fit(text: str, n_tokens, max_tokens: int) -> list[str]:
    """Coupe un morceau en deux (récursivement) tant qu'il dépasse la fenêtre du modèle.

    all-MiniLM-L6-v2 a max_seq_length = 256 : au-delà, la fin du morceau serait ignorée
    (troncature silencieuse). On coupe au milieu, sur un espace si possible.
    """
    if len(text) < 2 or n_tokens(text) <= max_tokens:
        return [text]
    mid = len(text) // 2
    cut = text.rfind(" ", 0, mid)
    cut = cut if cut > 0 else mid
    return split_to_fit(text[:cut], n_tokens, max_tokens) + split_to_fit(
        text[cut:], n_tokens, max_tokens
    )


def chunk_text(text: str, n_tokens, max_tokens: int) -> list[str]:
    """Morceaux de CHUNK_CHARS caractères, redécoupés si l'un dépasse max_tokens."""
    parts = [text[j : j + CHUNK_CHARS] for j in range(0, len(text), CHUNK_CHARS)] or [""]
    return [q for part in parts for q in split_to_fit(part, n_tokens, max_tokens)]


def embed_sections(texts: list[str]) -> np.ndarray:
    """Embedding d'un document = moyenne des embeddings de ses morceaux (aucun tronqué)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMB_MODEL, device="cpu")
    max_tokens = model.max_seq_length  # 256, jetons spéciaux [CLS]/[SEP] compris

    def n_tokens(t: str) -> int:
        return len(model.tokenizer(t)["input_ids"])

    chunks, owner = [], []
    for i, t in enumerate(texts):
        parts = chunk_text(t, n_tokens, max_tokens)
        chunks += parts
        owner += [i] * len(parts)
    assert max(n_tokens(c) for c in chunks) <= max_tokens  # garde : aucune troncature
    print(f"Morceaux : {len(chunks)} pour {len(texts)} documents", flush=True)
    vecs = model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    owner = np.asarray(owner)
    return np.vstack([vecs[owner == i].mean(axis=0) for i in range(len(texts))])


def get_embeddings(df: pd.DataFrame) -> np.ndarray:
    ids_path = EMB_CACHE.with_suffix(".ids.json")
    if EMB_CACHE.exists() and ids_path.exists():
        ids = json.loads(ids_path.read_text())
        if ids == df.index.tolist():
            return np.load(EMB_CACHE)
    emb = embed_sections(df.sections.tolist())
    np.save(EMB_CACHE, emb)
    ids_path.write_text(json.dumps(df.index.tolist()))
    return emb


# ------------------------------------------------------------------ modèles candidats
def tune_tfidf(texts_tr, y_tr, texts_va, y_va):
    best = None
    for c in C_GRID:
        m = md.make_m1_text(C=c).fit(texts_tr, y_tr)
        p = m.predict_proba(texts_va)[:, 1]
        s = average_precision_score(y_va, p)
        if best is None or s > best[0]:
            best = (s, c, m, p)
    return best


def make_emb_model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def tune_emb(x_tr, y_tr, x_va, y_va):
    best = None
    for c in EMB_C_GRID:
        m = make_emb_model(c).fit(x_tr, y_tr)
        p = m.predict_proba(x_va)[:, 1]
        s = average_precision_score(y_va, p)
        if best is None or s > best[0]:
            best = (s, c, m, p)
    return best


def stack_features(p_m1, p_text):
    return np.column_stack([logit(p_m1), logit(p_text)])


# ------------------------------------------------------------------ étape 1
def stage_select() -> None:
    df = load()
    tr, va = df.split == "train", df.split == "valid"
    y = df.y_fail.astype(int)
    rows, valid_preds = [], {}

    s = average_precision_score(y[va], df.m1_text[va])
    rows.append({"candidate": "m1_text_full", "valid_auc_pr": s, "params": "phase 2 (C = 1)"})
    valid_preds["m1_text_full"] = df.m1_text[va].to_numpy()

    for name, col in [("tfidf_sections", "sections"), ("tfidf_masked", "sections_masked")]:
        s, c, _, p = tune_tfidf(df[col][tr].tolist(), y[tr], df[col][va].tolist(), y[va])
        rows.append({"candidate": name, "valid_auc_pr": s, "params": {"C": c}})
        valid_preds[name] = p
        print(name, round(s, 4), c, flush=True)

    emb = get_embeddings(df)
    s, c, _, p = tune_emb(emb[tr.to_numpy()], y[tr], emb[va.to_numpy()], y[va])
    rows.append({"candidate": "emb_sections", "valid_auc_pr": s, "params": {"C": c}})
    valid_preds["emb_sections"] = p
    print("emb_sections", round(s, 4), c, flush=True)

    text_rows = [r for r in rows if r["candidate"] != "tfidf_masked"]  # masqué = diagnostic
    best_text = max(text_rows, key=lambda r: r["valid_auc_pr"])["candidate"]
    x_stack = stack_features(df.m1[va], valid_preds[best_text])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    p_cv = cross_val_predict(
        LogisticRegression(C=1.0), x_stack, y[va], cv=cv, method="predict_proba"
    )[:, 1]
    s = average_precision_score(y[va], p_cv)
    rows.append(
        {
            "candidate": "stack",
            "valid_auc_pr": s,
            "params": {"text_model": best_text, "C": 1.0, "cv": "5 plis dans la validation"},
        }
    )
    s_m1 = average_precision_score(y[va], df.m1[va])
    rows.append(
        {"candidate": "m1 (référence, sans texte)", "valid_auc_pr": s_m1, "params": "phase 2"}
    )

    candidates = [
        r
        for r in rows
        if r["candidate"] in ("m1_text_full", "tfidf_sections", "emb_sections", "stack")
    ]
    chosen = max(candidates, key=lambda r: r["valid_auc_pr"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "selection_validation.csv", index=False)
    params = {r["candidate"]: r["params"] for r in rows}
    FROZEN.write_text(
        json.dumps(
            {
                "chosen": chosen["candidate"],
                "params": params,
                "best_text_for_stack": best_text,
                "valid_scores": {r["candidate"]: r["valid_auc_pr"] for r in rows},
                "n_train": int(tr.sum()),
                "n_valid": int(va.sum()),
                "valid_failure_rate": float(y[va].mean()),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print("Figé :", chosen["candidate"])


# ------------------------------------------------------------------ étape 2
def fit_text_model(name, params, df, tr):
    """Renvoie une fonction de prédiction (sur un masque de lignes) pour un modèle de texte."""
    y = df.y_fail.astype(int)
    if name == "m1_text_full":
        return lambda mask: df.m1_text[mask].to_numpy()
    if name in ("tfidf_sections", "tfidf_masked"):
        col = "sections" if name == "tfidf_sections" else "sections_masked"
        m = md.make_m1_text(C=params[name]["C"]).fit(df[col][tr].tolist(), y[tr])
        return lambda mask: m.predict_proba(df[col][mask].tolist())[:, 1]
    if name == "emb_sections":
        emb = get_embeddings(df)
        m = make_emb_model(params[name]["C"]).fit(emb[tr.to_numpy()], y[tr])
        return lambda mask: m.predict_proba(emb[mask.to_numpy()])[:, 1]
    raise ValueError(name)


def stage_evaluate() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    df = load()
    tr, va, te = df.split == "train", df.split == "valid", df.split == "test"
    y = df.y_fail.astype(int)
    params = frozen["params"]
    chosen = frozen["chosen"]

    def predict(name):
        if name == "stack":
            text = fit_text_model(frozen["best_text_for_stack"], params, df, tr)
            stacker = LogisticRegression(C=1.0).fit(stack_features(df.m1[va], text(va)), y[va])
            return {
                s: stacker.predict_proba(stack_features(df.m1[m], text(m)))[:, 1]
                for s, m in [("valid", va), ("test", te)]
            }
        f = fit_text_model(name, params, df, tr)
        return {"valid": f(va), "test": f(te)}

    preds = {
        "chosen": predict(chosen),
        "tfidf_masked": predict("tfidf_masked"),
        "m1": {"valid": df.m1[va].to_numpy(), "test": df.m1[te].to_numpy()},
        "m1_text": {"valid": df.m1_text[va].to_numpy(), "test": df.m1_text[te].to_numpy()},
    }
    yt = y[te].to_numpy()
    res = {
        "chosen": chosen,
        "n_test": int(te.sum()),
        "test_failure_rate": float(yt.mean()),
        "models": {},
        "diffs": {},
    }
    calibrated = {}
    for name, p in preds.items():
        cal = PlattCalibrator().fit(y[va], p["valid"])
        pc = cal.transform(p["test"])
        calibrated[name] = pc
        ci = bootstrap_ci(yt, pc, N_BOOT, seed=11)
        res["models"][name] = {
            **ci,
            "ece_before": ece(yt, p["test"]) if name != "stack" else None,
            "ece_after": ece(yt, pc),
            "brier_before": compute_metrics(yt, p["test"])["brier"],
        }
    for other in ["m1", "m1_text"]:
        res["diffs"][f"chosen - {other}"] = bootstrap_diff(
            yt, calibrated["chosen"], calibrated[other], N_BOOT, seed=12
        )
    res["diffs"]["tfidf_masked - m1_text"] = bootstrap_diff(
        yt, calibrated["tfidf_masked"], calibrated["m1_text"], N_BOOT, seed=13
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "project_id": df.index[te],
            "y_fail": yt,
            **{f"p_fail_{k}": v for k, v in calibrated.items()},
        }
    ).to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
    res["portfolio"] = portfolio_backtest(df[te], calibrated["chosen"])
    # Contrôle de la sélection (évaluation, pas réglage) : même objectif figé, mêmes projets,
    # autres probabilités. M0 = probabilité constante (taux d'échec de l'entraînement) :
    # l'objectif ne voit alors que la taille des projets. M1 = sans texte.
    res["portfolio_control"] = {
        "m0": portfolio_backtest(df[te], np.full(int(te.sum()), float(y[tr].mean()))),
        "m1": portfolio_backtest(df[te], calibrated["m1"]),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(frozen, res)


def portfolio_backtest(test_df: pd.DataFrame, p_fail: np.ndarray) -> list[dict]:
    """Même protocole que le bloc A (objectif figé, 1 000 tirages, graine 2), autres probas."""
    from investagents.selection import choose_objective as co

    frozen_obj = json.loads(co.FROZEN.read_text(encoding="utf-8"))
    pool = co.load_pool("test")
    p_map = pd.Series(p_fail, index=test_df.index)
    pool = pool[pool.index.isin(p_map.index)].copy()
    pool["p_success"] = 1 - p_map.loc[pool.index].to_numpy()
    out = []
    for frac in co.BUDGETS:
        rand = co.random_draws(pool, frac, co.N_RANDOM_TEST, seed=2)
        chosen = co.select_all_cohorts(pool, frac, frozen_obj["objective"], frozen_obj["p_min"])
        m = co.portfolio_metrics(chosen)
        out.append(
            {
                "budget": frac,
                **m,
                **co.bootstrap_ci(chosen),
                "n_pool": len(pool),
                "random_count": float(rand.count_share_satisfactory.mean()),
                "random_budget": float(rand.budget_share_satisfactory.mean()),
                "p_count_vs_random": float(
                    (1 + (rand.count_share_satisfactory >= m["count_share_satisfactory"]).sum())
                    / 1001
                ),
                "p_budget_vs_random": float(
                    (1 + (rand.budget_share_satisfactory >= m["budget_share_satisfactory"]).sum())
                    / 1001
                ),
            }
        )
    return out


def fmt(m):
    return f"{m['value']:.3f} [{m['ci_low']:.3f} ; {m['ci_high']:.3f}]"


def write_report(frozen, res) -> None:
    sel = pd.read_csv(OUT_DIR / "selection_validation.csv")
    lines = [
        "# Bloc B : variantes de modèles de texte (généré automatiquement)",
        "",
        "Entraînement ≤ 2007 ; choix sur la validation 2008–2011 ; test 2012–2017 calculé une",
        "seule fois.",
        "Mêmes projets pour tous les modèles (ceux qui ont un texte). Classe positive = échec.",
        "",
        f"## 1. Validation ({frozen['n_valid']} projets, taux d'échec "
        f"{frozen['valid_failure_rate']:.1%}) — sert au choix",
        "",
        "| Candidat | AUC-PR validation | Réglage |",
        "| --- | ---: | --- |",
        *[f"| {r.candidate} | {r.valid_auc_pr:.3f} | {r.params} |" for r in sel.itertuples()],
        "",
        f"**Modèle figé : `{frozen['chosen']}`.** `tfidf_masked` est un diagnostic",
        "(non candidat) : masquage v2 = spaCy + listes fixes de pays et de capitales",
        "(GeoNames, accents ignorés). Limite : environ 1 grande ville par document reste",
        "visible (58 % des mentions de villes ≥ 100 000 hab. hors capitales ; mesure sur 300",
        "projets d'entraînement, reports/masquage_lieux.md) ; l'information pays est réduite,",
        "pas supprimée.",
        "Embeddings : morceaux redécoupés pour tenir dans 256 jetons (aucune troncature).",
        "",
        "le score de `stack` est estimé par validation croisée à 5 plis dans la validation.",
        "",
        f"## 2. Test ({res['n_test']} projets, taux d'échec {res['test_failure_rate']:.1%})",
        "",
        "Probabilités recalibrées (Platt) sur la validation.",
        "",
        "| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier après | ECE avant → après |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for name, m in res["models"].items():
        before = "—" if m["ece_before"] is None else f"{m['ece_before']:.3f}"
        lines.append(
            f"| {name} | {fmt(m['auc_pr'])} | {fmt(m['auc_roc'])} | "
            f"{m['brier']['value']:.4f} | {before} → {m['ece_after']:.3f} |"
        )
    lines += [
        "",
        "| Écart apparié | AUC-PR | AUC-ROC | Brier |",
        "| --- | --- | --- | --- |",
        *[
            f"| {k} | {fmt(d['auc_pr'])} | {fmt(d['auc_roc'])} | {fmt(d['brier'])} |"
            for k, d in res["diffs"].items()
        ],
        "",
        "## 3. Backtest de portefeuille avec le modèle figé (objectif figé du bloc A)",
        "",
        "| Budget | % projets satisfaisants [IC 95 %] | hasard | p | % budget → satisfaisants "
        "[IC 95 %] | hasard | p | Projets |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for r in res["portfolio"]:
        lines.append(
            f"| {r['budget']:.0%} | {r['count_share_satisfactory']:.1%} "
            f"[{r['count_share_ci'][0]:.1%} ; {r['count_share_ci'][1]:.1%}] | "
            f"{r['random_count']:.1%} | {r['p_count_vs_random']:.3f} | "
            f"{r['budget_share_satisfactory']:.1%} [{r['budget_share_ci'][0]:.1%} ; "
            f"{r['budget_share_ci'][1]:.1%}] | {r['random_budget']:.1%} | "
            f"{r['p_budget_vs_random']:.3f} | {r['n_projects']} |"
        )
    lines += [
        "",
        "## 4. Contrôle de la sélection : même objectif, probabilités de M0 et de M1",
        "",
        "M0 = probabilité constante (seule la taille des projets compte) ; M1 = sans texte.",
        "",
        "| Budget | Modèle | % projets satisfaisants | hasard | p | % budget → satisfaisants "
        "| hasard | p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    runs = {"figé": res["portfolio"], **res["portfolio_control"]}
    for i, base in enumerate(res["portfolio"]):
        for name, rows in runs.items():
            r = rows[i]
            lines.append(
                f"| {base['budget']:.0%} | {name} | {r['count_share_satisfactory']:.1%} | "
                f"{r['random_count']:.1%} | {r['p_count_vs_random']:.3f} | "
                f"{r['budget_share_satisfactory']:.1%} | {r['random_budget']:.1%} | "
                f"{r['p_budget_vs_random']:.3f} |"
            )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "evaluate"], required=True)
    args = parser.parse_args()
    stage_select() if args.stage == "select" else stage_evaluate()


if __name__ == "__main__":
    main()
