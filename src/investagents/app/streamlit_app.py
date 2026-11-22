"""Démo InvestAgents (MVP) — Streamlit.

Page « Portefeuille » : choix du budget (en % du montant demandé par cohorte) -> portefeuille
proposé par le sac à dos (OR-Tools) avec les probabilités calibrées de M1-texte, et métriques du
backtest. Page « Projets analysés » : arguments, 5 notes du Juge et citations vérifiées.

Lancer : `uv run streamlit run src/investagents/app/streamlit_app.py`
Lit uniquement des résultats déjà calculés (experiments/mvp/) : aucun appel à un LLM.
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
    "Page", ["Portefeuille", "Sélection multicritère (comité)", "Projets analysés"]
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
    from investagents.selection.multicriteria import scores_table, weighted_score

    st.header("Sélection multicritère à partir des notes du comité")
    st.warning(
        "Les notes du Juge servent à **classer les projets selon les préférences du décideur** "
        "(poids des 5 critères). Elles **ne prédisent pas l'échec** : AUC ≈ 0,5 sur la "
        "validation, et M3 / M4 ne font pas mieux que le hasard / que le texte seul au test "
        "(reports/blocC_m3_m4.md). Pour prédire l'échec, voir la page Portefeuille."
    )
    results_path = ROOT / "experiments" / "blocC" / "kaggle_run1" / "results.jsonl"
    sample_path = ROOT / "experiments" / "blocC" / "sample.csv"
    if not results_path.exists():
        st.warning("Résultats du comité absents : voir kaggle/blocC/README.md")
        st.stop()
    table = scores_table(results_path, sample_path)
    table = table[(table.split == "valid") & table.amount_at_approval_usd.notna()].copy()
    st.caption(
        f"{len(table)} projets de la validation (2008–2011) notés par le comité (Qwen2.5 7B). "
        "Le test n'est pas proposé ici : choisir des poids en regardant le test reviendrait à "
        "régler sur le test."
    )
    cols = st.columns(5)
    weights = {c: cols[i].slider(c, 0, 5, 1, key=f"w_{c}") for i, c in enumerate(CRITERIA)}
    if sum(weights.values()) == 0:
        st.error("Mettre au moins un poids non nul.")
        st.stop()
    frac = st.select_slider(
        "Budget (part du montant total des projets)",
        options=[0.10, 0.25, 0.50],
        value=0.25,
        format_func=lambda f: f"{f:.0%}",
    )
    table["score"] = weighted_score(table, weights)
    budget = frac * table.amount_at_approval_usd.sum()
    x = select_portfolio(
        table.score, table.amount_at_approval_usd, table.country, budget, 0.25, objective="p"
    )
    chosen = table[x].sort_values("score", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Projets choisis", f"{len(chosen)} / {len(table)}")
    c2.metric("Score moyen (0–1)", f"{chosen.score.mean():.2f}")
    c3.metric("Budget utilisé", f"{chosen.amount_at_approval_usd.sum() / 1e6:,.0f} M USD")
    st.caption(
        "Score = moyenne pondérée des 5 notes / 10. Sac à dos OR-Tools : maximise la somme des "
        "scores sous budget, au plus 25 % du budget par pays."
    )
    show = chosen[
        [
            "key",
            "country",
            "global_practice",
            "amount_at_approval_usd",
            "score",
            *CRITERIA,
            "outcome",
        ]
    ].rename(
        columns={"amount_at_approval_usd": "montant (USD)", "outcome": "note IEG (a posteriori)"}
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

else:
    st.header("Projets analysés par le comité (qwen2.5:3b, local)")
    files = sorted((MVP / "committee").glob("*.json"))
    if not files:
        st.warning("Lancer d'abord : uv run python -m investagents.agents.run_pilot")
        st.stop()
    data = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files}
    label = st.selectbox(
        "Projet", list(data), format_func=lambda k: f"{data[k]['masked_id']} ({k})"
    )
    d = data[label]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temps d'analyse", f"{d['seconds_total'] / 60:.1f} min")
    c2.metric("Citations valides", f"{d['citation_valid_rate']:.0%}")
    c3.metric("Secteur", d["global_practice"] or "—")
    c4.metric("Note IEG réelle", d["outcome_ieg"])
    if d["errors"]:
        st.error("Sortie invalide signalée : " + " ; ".join(d["errors"]))
    if d.get("judge"):
        scores = pd.DataFrame(d["judge"]["scores"]).set_index("criterion")
        st.subheader("Notes du Juge (0 à 10)")
        st.bar_chart(scores["score"])
        st.table(scores)
    if d.get("debate"):
        st.subheader("Arguments et citations")
        args = pd.DataFrame(d["debate"]["arguments"])
        args["phrases citées"] = [" / ".join(c["cited_text"]) for c in d["citations"]]
        args["citation valide"] = ["oui" if c["valid"] else "NON" for c in d["citations"]]
        st.dataframe(args, use_container_width=True)
    with st.expander("Texte donné aux agents (sections clés, identifiants masqués)"):
        for k, v in d["sections"].items():
            st.markdown(f"**{k}**")
            st.write(v)
