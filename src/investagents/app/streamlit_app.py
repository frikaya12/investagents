"""Démo InvestAgents (MVP) — Streamlit.

Page « Portefeuille » : choix du budget (en % du montant demandé par cohorte) -> portefeuille
proposé par le sac à dos (OR-Tools) avec les probabilités calibrées de M1-texte, et métriques du
backtest. Page « Sélection multicritère » : curseur α entre probabilité de réussite (ML) et notes
du comité, poids des 5 critères. Page « Fiche projet » : probabilité de réussite, arguments Pour /
Contre avec phrases citées, 5 notes du Juge (et facteurs du bloc D quand ils existent).

Lancer : `uv run streamlit run src/investagents/app/streamlit_app.py`
Lit uniquement des résultats déjà calculés (experiments/) : aucun appel à un LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from investagents.selection.knapsack import select_portfolio

ROOT = Path(__file__).resolve().parents[3]
MVP = ROOT / "experiments" / "mvp"
BLOC_A = ROOT / "experiments" / "blocA"

st.set_page_config(page_title="InvestAgents — démo", layout="wide")
st.title("InvestAgents — démo MVP")
st.caption(
    "Projets Banque mondiale approuvés en 2012–2017 (jeu de test). Données IEG © Banque "
    "mondiale, CC BY 4.0. Démo pédagogique, pas un conseil d'investissement."
)

page = st.sidebar.radio(
    "Page", ["Portefeuille", "Sélection multicritère (comité)", "Fiche projet"]
)


@st.cache_data
def load_backtest():
    path = MVP / "backtest.json"
    pool = MVP / "backtest_pool.parquet"
    if not path.exists() or not pool.exists():
        return None, None
    return json.loads(path.read_text(encoding="utf-8")), pd.read_parquet(pool)


if page == "Portefeuille":
    results, pool = load_backtest()
    if results is None:
        st.warning("Lancer d'abord : uv run python -m investagents.selection.run_backtest")
        st.stop()

    st.header("Sélection sous budget")
    col1, col2 = st.columns(2)
    frac = col1.select_slider(
        "Budget (part du montant total demandé par la cohorte)",
        options=[0.10, 0.25, 0.50],
        value=0.25,
        format_func=lambda f: f"{f:.0%}",
    )
    years = sorted(pool.approval_fy.unique())
    year = col2.selectbox("Cohorte (exercice d'approbation)", years, index=len(years) - 1)
    cohort = pool[pool.approval_fy == year].copy()
    budget = frac * cohort.cost.sum()
    frozen_path = BLOC_A / "objective_frozen.json"
    objective = (
        json.loads(frozen_path.read_text(encoding="utf-8"))["objective"]
        if frozen_path.exists()
        else "p"
    )
    x = select_portfolio(
        1 - cohort.m1_text_cal,
        cohort.cost,
        cohort.country,
        budget,
        results["max_country_share"],
        objective=objective,
    )
    chosen = cohort[x].copy()
    chosen["proba_succès"] = 1 - chosen.m1_text_cal
    chosen["montant (M USD)"] = chosen.cost / 1e6

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Budget", f"{budget / 1e6:,.0f} M USD")
    m2.metric("Projets financés", f"{len(chosen)} / {len(cohort)}")
    m3.metric("Succès attendu (modèle)", f"{chosen['proba_succès'].mean():.0%}")
    m4.metric("Réellement satisfaisants (IEG)", f"{(1 - chosen.y_fail).mean():.0%}")
    st.caption(
        f"Contrainte : au plus {results['max_country_share']:.0%} du budget par pays. "
        "La note IEG réelle n'est connue qu'après le projet : elle sert ici à vérifier le choix."
    )
    st.dataframe(
        chosen[["country", "global_practice", "montant (M USD)", "proba_succès", "outcome"]]
        .sort_values("proba_succès", ascending=False)
        .rename(columns={"country": "pays", "global_practice": "secteur", "outcome": "note IEG"}),
        use_container_width=True,
        hide_index=False,
    )

    st.caption(f"Objectif de sélection (choisi sur la validation) : `{objective}`.")
    obj_test = BLOC_A / "objective_test.json"
    if obj_test.exists():
        st.header("Objectif figé contre le hasard (test 2012–2017, calculé une fois)")
        rows = []
        for r in json.loads(obj_test.read_text(encoding="utf-8"))["results"]:
            for label, name in [
                ("chosen", f"{r['chosen']['objective']} (figé)"),
                ("initial_p", "p (initial)"),
            ]:
                m = r[label]
                rows.append(
                    {
                        "budget": f"{r['budget']:.0%}",
                        "objectif": name,
                        "% budget → satisfaisants": f"{m['budget_share_satisfactory']:.1%}",
                        "p (budget) vs hasard": f"{m['p_value_budget_vs_random']:.3f}",
                        "% projets satisfaisants": f"{m['count_share_satisfactory']:.1%}",
                        "p (projets) vs hasard": f"{m['p_value_count_vs_random']:.3f}",
                    }
                )
            rc = r["random"]["count_share_satisfactory"]["mean"]
            rb = r["random"]["budget_share_satisfactory"]["mean"]
            rows.append(
                {
                    "budget": f"{r['budget']:.0%}",
                    "objectif": "hasard (1 000)",
                    "% budget → satisfaisants": f"{rb:.1%}",
                    "p (budget) vs hasard": "—",
                    "% projets satisfaisants": f"{rc:.1%}",
                    "p (projets) vs hasard": "—",
                }
            )
        st.table(pd.DataFrame(rows))

    st.header("Backtest initial, objectif « p » (toutes cohortes 2012–2017)")
    rows = []
    for k, r in results["backtest"].items():
        rows.append(
            {
                "budget": k,
                "M1-texte": f"{r['m1_text']['share_satisfactory']:.1%} "
                f"[{r['m1_text']['ci_low']:.1%} ; {r['m1_text']['ci_high']:.1%}]",
                "M1": f"{r['m1']['share_satisfactory']:.1%} "
                f"[{r['m1']['ci_low']:.1%} ; {r['m1']['ci_high']:.1%}]",
                "hasard (1 000 tirages)": f"{r['random']['share_satisfactory']:.1%} "
                f"[{r['random']['p2_5']:.1%} ; {r['random']['p97_5']:.1%}]",
                "taux de base": f"{r['base_rate']:.1%}",
                "p vs hasard": f"{r['p_value_vs_random']:.3f}",
            }
        )
    st.table(pd.DataFrame(rows).set_index("budget"))
    st.subheader("Calibration (test)")
    st.table(
        pd.DataFrame(results["calibration"])
        .T[["test_brier_before", "test_brier_after", "test_ece_before", "test_ece_after"]]
        .rename(
            columns={
                "test_brier_before": "Brier avant",
                "test_brier_after": "Brier après",
                "test_ece_before": "ECE avant",
                "test_ece_after": "ECE après",
            }
        )
        .astype(float)
        .round(4)
    )

elif page == "Sélection multicritère (comité)":
    from investagents.schemas.committee import CRITERIA
    from investagents.selection.frontier import combined_score, committee_score, load_pool, select

    st.header("Décision multicritère : probabilité de réussite (ML) + notes du comité")
    st.warning(
        "Les notes du Juge servent à **classer les projets selon les préférences du décideur** "
        "(poids des 5 critères). Elles **ne prédisent pas l'échec** : AUC ≈ 0,5 sur la "
        "validation, et M3 / M4 ne font pas mieux que le hasard / que le texte seul au test "
        "(reports/blocC_m3_m4.md). La probabilité de réussite vient du ML (M1-texte calibré)."
    )
    if not (ROOT / "experiments" / "blocC" / "scores.parquet").exists():
        st.warning("Résultats du comité absents : voir kaggle/blocC/README.md")
        st.stop()
    pool = load_pool(
        "valid", ROOT / "experiments/blocC/scores.parquet", MVP / "predictions.parquet"
    )
    st.caption(
        f"{len(pool)} projets de la validation (2008–2011) notés par le comité (Qwen2.5 7B). "
        "Le test n'est pas proposé ici : choisir α ou des poids en regardant le test reviendrait "
        "à régler sur le test. Frontière complète : reports/frontiere_alpha.md."
    )
    alpha = st.slider(
        "α : poids de la probabilité de réussite (1 = ML seul, 0 = comité seul)",
        0.0, 1.0, 0.5, 0.05,
    )  # fmt: skip
    cols = st.columns(5)
    weights = {c: cols[i].slider(c, 0, 5, 1, key=f"w_{c}") for i, c in enumerate(CRITERIA)}
    if sum(weights.values()) == 0:
        st.error("Mettre au moins un poids non nul.")
        st.stop()
    frac = st.select_slider(
        "Budget (part du montant de chaque cohorte annuelle)",
        options=[0.10, 0.25, 0.50],
        value=0.25,
        format_func=lambda f: f"{f:.0%}",
    )
    pool["score"] = combined_score(pool, alpha, weights)
    pool["note_comite"] = committee_score(pool, weights)
    chosen = select(pool, pool.score, frac).sort_values("score", ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projets choisis", f"{len(chosen)} / {len(pool)}")
    c2.metric(
        "Réussite attendue (ML)",
        f"{chosen.p_success.mean():.1%}",
        f"{100 * (chosen.p_success.mean() - pool.p_success.mean()):+.1f} pts vs ensemble",
    )
    c3.metric("Note moyenne du comité (0–1)", f"{chosen.note_comite.mean():.2f}")
    c4.metric("Budget utilisé", f"{chosen.cost.sum() / 1e6:,.0f} M USD")
    st.subheader("Profil du portefeuille sur les 5 critères (notes sur 10)")
    prof = pd.DataFrame(
        {
            "portefeuille": chosen[list(CRITERIA)].mean(),
            "tous les projets": pool[list(CRITERIA)].mean(),
        }
    )
    st.bar_chart(prof, stack=False)
    st.caption(
        "score = α × p_succès + (1 − α) × moyenne pondérée des notes / 10. Sac à dos OR-Tools "
        "(objectif figé Σ score × montant), une sélection par cohorte annuelle, au plus 25 % du "
        "budget par pays. La réussite attendue est la moyenne des probabilités du ML, pas l'issue."
    )
    show = chosen[
        ["key", "country", "global_practice", "cost", "p_success", "score", *CRITERIA, "outcome"]
    ].rename(
        columns={"cost": "montant (USD)", "p_success": "p réussite (ML)",
                 "outcome": "note IEG (a posteriori)"}
    )  # fmt: skip
    st.dataframe(show, use_container_width=True, hide_index=True)

else:
    from investagents.app.project_sheet import load_all, sheet
    from investagents.schemas.committee import CRITERIA

    st.header("Fiche projet : ML + agents")
    if not (ROOT / "experiments/blocC/kaggle_run1/results.jsonl").exists():
        st.warning("Résultats du comité absents : voir kaggle/blocC/README.md")
        st.stop()
    recs, meta = load_all()
    split = st.radio("Période", ["valid", "test"], horizontal=True,
                     format_func=lambda s: {"valid": "validation 2008–2011",
                                            "test": "test 2012–2017"}[s])  # fmt: skip
    sub = meta[meta.split == split]
    key = st.selectbox(
        "Projet (clé masquée)",
        sub.key.tolist(),
        format_func=lambda k: f"{k} — {sub.set_index('key').loc[k, 'country']}, "
        f"{sub.set_index('key').loc[k, 'global_practice']}",
    )
    s = sheet(recs[key], sub[sub.key == key].iloc[0])
    mean_note = pd.Series([x["score"] for x in s["scores"]]).mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilité de réussite (ML)", f"{s['p_success']:.0%}")
    c2.metric("Note moyenne du comité", f"{mean_note:.1f} / 10")
    c3.metric("Montant", f"{(s['amount_usd'] or 0) / 1e6:,.0f} M USD")
    c4.metric("Secteur", s["sector"] or "—")
    st.caption(
        "La probabilité vient du ML (M1-texte calibré). Les notes et arguments viennent du comité "
        "(Qwen2.5 7B) : ils expliquent et classent selon les critères ; ils ne prédisent pas "
        "l'échec."
    )
    scores = pd.DataFrame(s["scores"]).set_index("criterion")
    st.subheader("Notes du Juge sur les 5 critères (0 à 10)")
    st.bar_chart(scores["score"])
    st.subheader("Arguments Pour et Contre, avec les phrases citées du dossier")
    for c in CRITERIA:
        st.markdown(f"**{c}** — note {scores.loc[c, 'score']} : {scores.loc[c, 'justification']}")
        cols = st.columns(2)
        for col, stance in zip(cols, ("pro", "contra"), strict=True):
            for a in [x for x in s["arguments"] if x["criterion"] == c and x["stance"] == stance]:
                label = "Pour" if stance == "pro" else "Contre"
                verdict = a["critic_verdict"] or "—"
                col.markdown(f"*{label}* (critique : {verdict}) — {a['claim']}")
                flag = "" if a["citation_valid"] else " (NON vérifiée)"
                for t in a["cited"]:
                    col.caption(f"« {t[:400]} »{flag}")
    if s["factors"]:
        st.subheader("Facteurs de risque extraits (bloc D)")
        st.dataframe(pd.DataFrame(s["factors"]).T[["status", "value", "sentence_ids"]])
    with st.expander("Note IEG réelle (a posteriori, jamais vue par les agents)"):
        st.write(s["outcome"])
