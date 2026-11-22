"""Évaluation finale UNIQUE sur le coffre-fort (modèles sans agents : M0, M1, M1-texte).

Coffre-fort : projets évalués par l'IEG en exercice ≥ 2025 (700 projets notés), jamais utilisés
jusqu'ici. Évaluation demandée par Aya en fin de projet.

Étapes :
1. `--stage protocol` : écrit experiments/vault/protocol.json AVANT de lire le coffre (modèles,
   chaîne de traitement, population, métriques, graine). Refuse d'écraser un protocole existant.
2. `--stage texts` : télécharge les dossiers ex ante des projets du coffre, choisis par
   leakage.py (même règle que le corpus : anglais, daté strictement avant l'approbation, le plus
   récent des PAD / SAR / Program Document) -> data/processed/vault_corpus_index.parquet.
3. `--stage evaluate` : calcul unique ; refuse de tourner deux fois (results.json existe).

Chaîne figée (la même que le test du bloc B et la sélection de portefeuille) : M1 et M1-texte
entraînés sur l'entraînement (≤ 2007) avec les hyperparamètres figés
(experiments/phase2/frozen.json), probabilités recalibrées (Platt) sur la validation 2008–2011.
M0 : taux d'échec de l'entraînement (constant).

Lancer : `uv run python -m investagents.ml.run_vault --stage protocol|texts|evaluate`
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from investagents.data.corpus import (
    CENSUS_PATH,
    MIN_CHARS,
    N_THREADS,
    PROJECTS_PATH,
    TEXT_DIR,
    download_gz,
    read_text,
    select_main_documents,
)
from investagents.data.leakage import filter_ex_ante
from investagents.ml import models as md
from investagents.ml.calibration import PlattCalibrator, ece
from investagents.ml.features import build_features, target_failure
from investagents.ml.metrics import bootstrap_ci, bootstrap_diff
from investagents.ml.run_phase2 import FROZEN_PATH, load_data, load_texts
from investagents.ml.split import load_modeling_table

OUT = Path("experiments/vault")
PROTOCOL = OUT / "protocol.json"
RESULTS = OUT / "results.json"
VAULT_INDEX = Path("data/processed/vault_corpus_index.parquet")
REPORT = Path("reports/coffre_fort.md")
N_BOOT = 2000
SEED = 31


def stage_protocol() -> None:
    if PROTOCOL.exists():
        raise SystemExit(f"Protocole déjà figé : {PROTOCOL}")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "population": "tous les projets du coffre-fort (évaluation IEG ≥ 2025) avec une note "
        "d'issue ; M1-texte sur ceux qui ont un dossier ex ante exploitable (≥ 10 000 car.)",
        "models": {
            "M0": "taux d'échec de l'entraînement (≤ 2007), constant",
            "M1": "gradient boosting, groupes base + wdi, hyperparamètres de "
            "experiments/phase2/frozen.json, entraîné sur ≤ 2007, Platt sur 2008–2011",
            "M1-texte": "TF-IDF + régression logistique C = 1, entraîné sur ≤ 2007, Platt sur "
            "2008–2011",
        },
        "metrics": ["auc_pr (classe échec)", "auc_roc", "brier", "ece (10 classes)"],
        "ci": f"bootstrap percentile 95 %, {N_BOOT} tirages, graine {SEED}",
        "paired": ["M1-texte − M1 (projets avec texte)", "M1 − M0"],
        "breakdown": "approbation ≤ 2017 / ≥ 2018 (descriptif, aucun choix n'en dépend)",
        "note": "Calcul unique ; aucun modèle, seuil ni réglage ne change après lecture.",
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Protocole figé : {PROTOCOL}")


def vault_table() -> pd.DataFrame:
    return load_modeling_table(with_country_indicators=True, vault=True)


def stage_texts() -> None:
    if not PROTOCOL.exists():
        raise SystemExit("Figer d'abord le protocole (--stage protocol).")
    table = vault_table()
    projects = pd.read_parquet(PROJECTS_PATH)[["project_id", "approval_date"]]
    approvals = table[["project_id", "approval_fy"]].merge(projects, on="project_id", how="left")
    census = pd.read_parquet(CENSUS_PATH)
    census = census[census["project_id"].isin(set(table["project_id"]))]
    docs = select_main_documents(filter_ex_ante(census, approvals))
    print(f"Coffre : {len(table)} projets, documents à télécharger : {len(docs)}", flush=True)
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        docs["n_chars"] = list(pool.map(lambda r: download_gz(r.doc_id, r.txturl),
                                        docs.itertuples()))  # fmt: skip
    docs["usable"] = docs["n_chars"] >= MIN_CHARS
    cols = ["project_id", "doc_id", "docty", "docdt", "approval_ref", "n_chars", "usable"]
    docs[cols].to_parquet(VAULT_INDEX, index=False)
    print(f"Textes exploitables : {int(docs.usable.sum())} ; écrit : {VAULT_INDEX}")


def stage_evaluate() -> None:
    if RESULTS.exists():
        raise SystemExit(f"Déjà évalué ({RESULTS}) : le coffre-fort ne se calcule qu'une fois.")
    if not PROTOCOL.exists() or not VAULT_INDEX.exists():
        raise SystemExit("Protocole et textes d'abord.")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    # données hors coffre (entraînement + validation) et coffre, mêmes variables
    table, y, split = load_data()
    vt = vault_table().reset_index(drop=True)
    yv = target_failure(vt)
    both = pd.concat([table, vt], ignore_index=True)  # mêmes catégories pour les deux
    x_all = build_features(both, tuple(frozen["m1"]["groups"]))
    x, xv = x_all.iloc[: len(table)], x_all.iloc[len(table) :]
    x.index, xv.index = y.index, yv.index
    tr, va = split == "train", split == "valid"

    m1 = md.make_m1(**frozen["m1"]["params"]).fit(x[tr], y[tr])
    cal1 = PlattCalibrator().fit(y[va], m1.predict_proba(x[va])[:, 1])
    p_m1 = cal1.transform(m1.predict_proba(xv)[:, 1])

    texts = load_texts(y.index[tr | va])
    has = pd.Series(y.index.isin(texts.index), index=y.index)
    mt = md.make_m1_text(**frozen["m1_text"]["params"])
    mt.fit(texts.loc[y.index[tr & has]].tolist(), y[tr & has])
    calt = PlattCalibrator().fit(
        y[va & has], mt.predict_proba(texts.loc[y.index[va & has]].tolist())[:, 1]
    )
    idx = pd.read_parquet(VAULT_INDEX)
    idx = idx[idx.usable]
    vtexts = pd.Series(
        {r.project_id: read_text(r.doc_id, TEXT_DIR)[: md.TEXT_MAX_CHARS] for r in idx.itertuples()}
    )
    with_text = yv.index.isin(vtexts.index)
    p_text = np.full(len(yv), np.nan)
    p_text[with_text] = calt.transform(
        mt.predict_proba(vtexts.loc[yv.index[with_text]].tolist())[:, 1]
    )
    p_m0 = np.full(len(yv), float(y[tr].mean()))

    def block(mask, label):
        yy = yv.to_numpy()[mask]
        out = {"label": label, "n": int(mask.sum()), "failure_rate": float(yy.mean()),
               "models": {}, "diffs": {}}  # fmt: skip
        for name, p in [("M0", p_m0), ("M1", p_m1)]:
            pp = p[mask]
            out["models"][name] = {
                "ci": bootstrap_ci(yy, pp, N_BOOT, seed=SEED) if name != "M0" else None,
                "brier": float(np.mean((pp - yy) ** 2)),
                "ece": ece(yy, pp),
            }
        out["diffs"]["M1 - M0"] = bootstrap_diff(yy, p_m1[mask], p_m0[mask], N_BOOT, SEED)
        mt_mask = mask & with_text
        yt = yv.to_numpy()[mt_mask]
        out["text_subset"] = {"n": int(mt_mask.sum()), "failure_rate": float(yt.mean())}
        for name, p in [("M1", p_m1), ("M1-texte", p_text)]:
            pp = p[mt_mask]
            out["text_subset"][name] = {
                "ci": bootstrap_ci(yt, pp, N_BOOT, seed=SEED),
                "brier": float(np.mean((pp - yt) ** 2)),
                "ece": ece(yt, pp),
            }
        out["text_subset"]["M1-texte - M1"] = bootstrap_diff(
            yt, p_text[mt_mask], p_m1[mt_mask], N_BOOT, SEED
        )
        return out

    fy = vt.approval_fy.astype(int).to_numpy()
    allm = np.ones(len(yv), dtype=bool)
    res = {
        "all": block(allm, "tout le coffre-fort"),
        "le2017": block(fy <= 2017, "approbation ≤ 2017"),
        "ge2018": block(fy >= 2018, "approbation ≥ 2018"),
        "n_vault": len(yv),
        "n_with_text": int(with_text.sum()),
    }
    RESULTS.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        {"project_id": yv.index, "approval_fy": fy, "y_fail": yv.to_numpy(), "p_m0": p_m0,
         "p_m1": p_m1, "p_m1_text": p_text}  # fmt: skip
    ).to_parquet(OUT / "predictions.parquet", index=False)
    write_report(res)


def fmt(d) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f} ; {d['ci_high']:.3f}]"


def write_report(res) -> None:
    lines = [
        "# Coffre-fort : évaluation finale unique, modèles sans agents (généré)",
        "",
        "Protocole figé avant lecture : experiments/vault/protocol.json. Modèles entraînés sur",
        "≤ 2007, recalibrés (Platt) sur 2008–2011, appliqués une seule fois aux projets évalués",
        f"par l'IEG en exercice ≥ 2025 ({res['n_vault']} projets, dont {res['n_with_text']}",
        "avec un dossier ex ante exploitable). Classe positive = échec.",
        "",
    ]
    for key in ("all", "le2017", "ge2018"):
        b = res[key]
        t = b["text_subset"]
        lines += [
            f"## {b['label']} ({b['n']} projets, taux d'échec {b['failure_rate']:.1%})",
            "",
            "| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |",
            "| --- | --- | --- | ---: | ---: |",
            f"| M0 | {b['failure_rate']:.3f} (= taux d'échec) | 0.500 | "
            f"{b['models']['M0']['brier']:.4f} | {b['models']['M0']['ece']:.3f} |",
            f"| M1 | {fmt(b['models']['M1']['ci']['auc_pr'])} | "
            f"{fmt(b['models']['M1']['ci']['auc_roc'])} | {b['models']['M1']['brier']:.4f} | "
            f"{b['models']['M1']['ece']:.3f} |",
            "",
            f"M1 − M0 : AUC-PR {fmt(b['diffs']['M1 - M0']['auc_pr'])}, Brier "
            f"{fmt(b['diffs']['M1 - M0']['brier'])}.",
            "",
            f"Projets avec texte ({t['n']}, taux d'échec {t['failure_rate']:.1%}) :",
            "",
            "| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for name in ("M1", "M1-texte"):
            m = t[name]
            lines.append(
                f"| {name} | {fmt(m['ci']['auc_pr'])} | {fmt(m['ci']['auc_roc'])} | "
                f"{m['brier']:.4f} | {m['ece']:.3f} |"
            )
        d = t["M1-texte - M1"]
        lines += [
            "",
            f"M1-texte − M1 : AUC-PR {fmt(d['auc_pr'])}, AUC-ROC {fmt(d['auc_roc'])}, "
            f"Brier {fmt(d['brier'])}.",
            "",
        ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["protocol", "texts", "evaluate"], required=True)
    stage = parser.parse_args().stage
    {"protocol": stage_protocol, "texts": stage_texts, "evaluate": stage_evaluate}[stage]()


if __name__ == "__main__":
    main()
