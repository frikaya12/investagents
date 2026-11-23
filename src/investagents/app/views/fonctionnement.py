"""Comment ça marche : chaque étape du pipeline, en simple puis en technique."""

import streamlit as st

from investagents.app import data, ui

t = data.metrics().txt

st.title("Comment ça marche")
ui.intro("Cliquez sur une étape pour voir ce qu'elle fait, en simple puis en technique.")

STEPS = {
    "1 · Données": (
        f"Nous avons pris {t['n_exploitable']} vrais projets de la Banque mondiale dont on "
        f"connaît la "
        "fin : l'évaluateur indépendant (IEG) les a notés après leur clôture. Pour chacun, on a "
        "le dossier écrit avant le financement.",
        f"Notes IEG (CC BY 4.0) ; cible failure = 1 si MU, U ou HU. Dossiers ex ante (PAD / SAR) "
        f"via la Documents & Reports API, ≥ 10 000 caractères. Taux d'échec : {t['fail_train']} "
        f"(≤ 2007), {t['fail_valid']} (2008–2011), {t['fail_test']} (2012–2017).",
        "IEG",
    ),
    "2 · Anti-fuite": (
        "Le modèle ne doit jamais voir l'avenir : seuls les documents écrits avant l'approbation "
        "sont lus, les noms sont masqués, et un paquet de projets reste sous clé jusqu'à la fin.",
        "Filtre unique data/leakage.py (date strictement antérieure, pas d'ICR / ICRR / PPAR) ; "
        "colonnes post-projet refusées par assert_no_forbidden ; masquage spaCy + GeoNames ; "
        f"test de reconnaissance : {t['rec_recognized']} projet reconnu ; split temporel et "
        "coffre-fort (évalués en 2025 ou après).",
        "fuite de données",
    ),
    "3 · Machine learning": (
        "Des modèles classiques apprennent sur le passé à repérer les projets risqués : l'un à "
        "partir de chiffres (région, secteur, montant…), l'autre à partir des mots du dossier.",
        f"M1 : gradient boosting (HistGradientBoosting) ; M1-texte : TF-IDF + régression "
        f"logistique, AUC-PR {t['m1t_aucpr']} au test. Probabilités recalibrées par Platt sur "
        f"la validation (ECE {t['ece_before']} → {t['ece_after']}).",
        "gradient boosting",
    ),
    "4 · Agents juges": (
        "Pour chaque projet, deux agents plaident Pour et Contre sur 5 critères en citant le "
        "dossier ; un critique juge leurs arguments ; un Juge note de 0 à 10.",
        f"LangGraph, Qwen2.5 7B via Ollama sur GPU T4 gratuits (Kaggle) ; sorties JSON Pydantic ; "
        f"prompts adaptés de DIALECTIC (MIT). {t['com_judged']} projets, citations exactes "
        f"{t['com_quotes_pct']}. M3 (notes seules) : {t['c_m3_aucpr']} contre {t['c_base']}.",
        "agent",
    ),
    "5 · Agents extracteurs": (
        "Un autre agent répond à 19 questions factuelles sur le dossier (réinstallation de "
        "populations ? agence à capacité faible ?), avec la phrase qui le prouve.",
        f"Grille figée de 19 facteurs, présélection des phrases en Python, citations par numéros, "
        f"comptes en Python. {t['d_full_n']} dossiers, citations valides {t['d_full_cit']}. "
        f"M5 − M0 = {t['m5_minus_m0']} ; M6 − texte = {t['m6_minus_text']}.",
        "citation vérifiée",
    ),
    "6 · Sélection": (
        "Avec un budget limité, un solveur choisit le panier de projets qui maximise les chances "
        "de réussite ; un curseur permet d'y mêler le jugement du comité.",
        f"Sac à dos exact OR-Tools CP-SAT, objectif figé Σ p × coût, une sélection par cohorte, "
        f"≤ 25 % du budget par pays. Test : {t['pf25']} de projets réussis contre "
        f"{t['pf25_random']} au hasard ({t['pf25_p']}). Frontière α : α × p + (1 − α) × notes.",
        "sac à dos (CP-SAT)",
    ),
}

step = st.pills("Étape", list(STEPS), default=list(STEPS)[0], label_visibility="collapsed")
step = step or list(STEPS)[0]
simple, tech, term = STEPS[step]
c1, c2 = st.columns(2)
with c1:
    st.markdown(f'<div class="ia-take" style="min-height:9rem"><b>En simple.</b> {simple}</div>',
                unsafe_allow_html=True)  # fmt: skip
with c2:
    st.markdown(ui.card("En technique", tech), unsafe_allow_html=True)
st.caption(f"Terme clé : {term}. " + ui.gloss(term))
