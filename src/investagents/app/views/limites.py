"""Limites et honnêteté : le même texte que la section 11 du rapport, et le coffre-fort en clair."""

import streamlit as st

from investagents.app import data, ui
from investagents.report.build import section_markdown
from investagents.report.metrics import num

m = data.metrics()
t, r = m.txt, m.raw

st.title("Limites et honnêteté")
ui.intro("Ce que ce projet ne permet pas de dire. Ces limites sont aussi dans le rapport final, "
         "mot pour mot.")

c1, c2, c3 = st.columns(3)
c1.metric("Coffre-fort · AUC-ROC de M1", num(r["vault_m1_aucroc"]["value"]),
          help=ui.gloss("coffre-fort"), border=True)  # fmt: skip
c1.caption("0,5 = hasard : sur des projets neufs, M1 ne prédit plus.")
c2.metric("Backtest : même modèle, deux pools", f"{t['pf25_share']} / {t['bt_text']}",
          border=True)  # fmt: skip
c2.caption(f"{t['bt_pool_gap']} projets d'écart suffisent à déplacer le résultat d'environ 5 "
           "points.")  # fmt: skip
c3.metric("Textes du coffre-fort téléchargés", t["vault_text_n"], border=True)
c3.caption(f"sur {t['vault_n']} projets : le serveur a refusé les téléchargements (403).")

st.markdown(section_markdown(11, m).split("**En technique.**", 1)[1])
