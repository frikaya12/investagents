"""Démo InvestAgents — point d'entrée Streamlit (navigation multipage).

Histoire guidée : Accueil → Fiche projet → Résultats, puis Comment ça marche, Décision
multicritère, Les agents, Limites, Glossaire (une page = un fichier de views/).

Lancer : `uv run streamlit run src/investagents/app/streamlit_app.py`
Lit uniquement des résultats déjà calculés (experiments/, via report/metrics.py) : aucun appel
à un LLM, aucune clé d'API, aucun calcul de modèle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:  # Streamlit Cloud : le paquet n'est pas installé
    sys.path.insert(0, str(ROOT / "src"))

from investagents.app import ui  # noqa: E402

REPORT_URL = "https://frikaya12.github.io/investagents/rapport.html"
REPO_URL = "https://github.com/frikaya12/investagents"

st.set_page_config(page_title="InvestAgents — démo", page_icon="📊", layout="wide")
ui.setup()

pages = {
    "L'histoire": [
        st.Page("views/accueil.py", title="Accueil", icon="🏠", default=True),
        st.Page("views/fiche.py", title="Fiche projet", icon="📄"),
        st.Page("views/resultats.py", title="Résultats", icon="📊"),
    ],
    "Pour aller plus loin": [
        st.Page("views/fonctionnement.py", title="Comment ça marche", icon="🔎"),
        st.Page("views/decision.py", title="Décision multicritère", icon="🎚️"),
        st.Page("views/agents.py", title="Les agents", icon="🤖"),
        st.Page("views/limites.py", title="Limites et honnêteté", icon="⚖️"),
        st.Page("views/glossaire.py", title="Glossaire", icon="📖"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.markdown(f"[Rapport final]({REPORT_URL}) · [Code]({REPO_URL})")
    st.caption("Projets Banque mondiale ; données IEG © Banque mondiale, CC BY 4.0. Démo "
               "pédagogique, pas un conseil d'investissement. LLM : Qwen2.5 7B via Ollama, 0 €.")
nav.run()
