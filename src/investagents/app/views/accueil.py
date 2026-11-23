"""Page Accueil : le problème en 10 secondes, 3 chiffres clés, pipeline, résumé."""

import streamlit as st

from investagents.app import data, ui
from investagents.report.metrics import num

m = data.metrics()
t, r = m.txt, m.raw

st.title("Des agents d'IA peuvent-ils aider à choisir les projets à financer ?")
ui.intro(
    "La Banque mondiale finance des milliers de projets de développement ; environ un sur quatre "
    "est jugé en échec une fois terminé. Nous avons fait lire à des agents d'IA les "
    "dossiers écrits "
    "<b>avant</b> le financement, puis vérifié leurs avis sur les vrais résultats."
)

c1, c2, c3 = st.columns(3)
c1.metric(
    "Projets réussis dans le portefeuille choisi",
    t["pf25_share"],
    help=ui.gloss("sac à dos (CP-SAT)"),
    border=True,
)
c1.caption(f"contre {t['pf25_random']} pour un portefeuille tiré au hasard ({t['pf25_p']}), "
           "même budget (25 %), test 2012–2017.")  # fmt: skip
c2.metric(
    "Le texte du dossier prédit l'échec… un peu",
    num(r["m1t_aucpr"]["value"]),
    help=ui.gloss("AUC-PR"),
    border=True,
)
c2.caption(f"AUC-PR de M1-texte, contre {t['base_test_text']} pour le hasard : signal réel mais "
           "faible.")  # fmt: skip
c3.metric(
    "Citations vérifiées mot pour mot",
    t["com_quotes_pct"],
    help=ui.gloss("citation vérifiée"),
    border=True,
)
c3.caption(f"{t['com_quotes']} citations vérifiées en Python ; {t['rec_recognized']} projet "
           "reconnu de mémoire par le LLM.")  # fmt: skip

st.markdown(
    '<div class="ia-message">« Le ML prédit faiblement ; '
    "les agents justifient chaque décision avec des sources vérifiées et extraient des "
    "facteurs de risque lisibles et prédictifs, mais "
    "n'ajoutent rien au texte complet. »</div>",
    unsafe_allow_html=True,
)

st.subheader("Le chemin d'un projet, du dossier à la décision")
steps = [
    ("1 · Données", f"{t['n_exploitable']} projets : note IEG + dossier écrit avant l'approbation"),
    ("2 · Anti-fuite", "documents datés avant l'approbation, noms masqués, coffre-fort"),
    ("3 · ML", "M1 (chiffres), M1-texte (mots du dossier) : une probabilité de réussite"),
    ("4 · Agents", "juges Pour / Contre sur 5 critères, extracteurs de 19 facteurs"),
    ("5 · Sélection", "sac à dos sous budget, curseur α entre ML et comité"),
]
arrow = '<div class="ia-arrow">→</div>'
st.markdown('<div class="ia-flow">' + arrow.join(ui.card(a, b) for a, b in steps) + "</div>",
            unsafe_allow_html=True)  # fmt: skip
st.page_link("views/fonctionnement.py", label="Voir chaque étape en détail", icon="🔎")

st.subheader("Le projet en 30 secondes")
st.markdown(
    f"""
- **La question :** un comité d'agents d'IA classe-t-il les projets mieux qu'un modèle de machine
  learning classique ?
- **La réponse pour prédire : non.** Le meilleur prédicteur est le texte du dossier lui-même ;
  les notes des juges ne prédisent pas l'échec, et rien de ce que font les agents n'ajoute au texte
  ({t['m6_minus_text']} d'AUC-PR avec les facteurs extraits).
- **La réponse pour décider : oui, pour expliquer.** Chaque note est justifiée par des phrases du
  dossier, vérifiées ; 19 facteurs de risque lisibles sont extraits avec leur source.
- **Sur des projets neufs (coffre-fort)**, le modèle à variables chiffrées tombe au niveau du
  hasard : AUC-ROC {t['vault_m1_aucroc']}.
"""
)

c1, c2 = st.columns(2)
c1.page_link("views/fiche.py", label="Suivre un vrai projet : la fiche projet", icon="📄")
c2.page_link("views/resultats.py", label="Voir tous les résultats", icon="📊")

with st.expander("Résumé de présentation : chaque étape, sa métrique et ce qu'elle veut dire"):
    st.markdown(data.resume_prof())
