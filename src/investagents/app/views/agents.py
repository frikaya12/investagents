"""Les agents : juges contre extracteurs, fiabilité, reconnaissance, poids des facteurs, apport."""

import plotly.graph_objects as go
import streamlit as st

from investagents.app import data, ui
from investagents.report.build import section_markdown
from investagents.report.metrics import chart_factor_weights

m = data.metrics()
t = m.txt

st.title("Les agents")
ui.intro("Deux familles d'agents, construites avec le même LLM (Qwen2.5 7B via Ollama) : des "
         "juges qui débattent et notent, des extracteurs qui relèvent des faits. Ce qu'ils font "
         "bien, et ce qu'ils n'apportent pas.")  # fmt: skip

c1, c2 = st.columns(2)
with c1:
    st.subheader("Les juges (bloc C)")
    a, b = st.columns(2)
    a.metric("Projets notés", t["com_judged"], border=True)
    b.metric("Temps par projet (GPU T4)", t["com_seconds"], border=True)
    a.metric("Citations exactes", t["com_quotes_pct"], help=ui.gloss("citation vérifiée"),
             border=True)  # fmt: skip
    b.metric("M3 · AUC-PR (notes seules)", t["c_m3_aucpr"].split(" ")[0],
             help=ui.gloss("AUC-PR"), border=True)  # fmt: skip
    st.caption(f"M3 contre un taux de base de {t['c_base']} : les notes ne prédisent pas "
               f"l'échec. M4 − texte {t['m4_minus_text']} ; M7 − texte {t['m7_minus_text']}.")
with c2:
    st.subheader("Les extracteurs (bloc D)")
    a, b = st.columns(2)
    a.metric("Dossiers lus", t["d_full_n"], border=True)
    b.metric("Temps par dossier (GPU T4)", t["d_full_s"], border=True)
    a.metric("Citations valides", t["d_full_cit"], help=ui.gloss("citation vérifiée"),
             border=True)  # fmt: skip
    b.metric("M5 − M0 · AUC-PR", t["m5_minus_m0"].split(" ")[0], help=ui.gloss("écart apparié"),
             border=True)  # fmt: skip
    st.caption(f"M5 − M0 = {t['m5_minus_m0']} : signal réel, faible. M6 − texte "
               f"{t['m6_minus_text']} : rien de plus que le texte.")
ui.takeaway("Les deux familles sont fiables (citations vérifiées, aucune erreur), mais seuls les "
            "facteurs extraits portent un signal prédictif, et ce signal est déjà dans le texte.")

st.subheader("Les agents trichent-ils avec leur mémoire ?")
c1, c2, c3 = st.columns(3)
c1.metric("Projets reconnus", t["rec_recognized"], help=ui.gloss("test de reconnaissance"),
          border=True)  # fmt: skip
c2.metric("Notes IEG proposées", t["rec_rating"], border=True)
c3.metric("Arguments réécrits au raffinement", t["args_rewritten"], border=True)
ui.takeaway("Sur des dossiers masqués, le LLM ne reconnaît aucun projet et ne propose aucune note "
            "IEG : ses avis viennent du dossier, pas de sa mémoire.")

st.subheader("Quels facteurs vont avec plus d'échecs ?")
w = chart_factor_weights(m)
fig = go.Figure(go.Bar(
    x=[x["weight"] for x in w], y=[x["label"] for x in w], orientation="h",
    marker={"color": [ui.ORANGE if x["weight"] > 0 else ui.BLUE for x in w]},
    hovertemplate="%{y} : %{x:+.4f}<extra></extra>",
))  # fmt: skip
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="Poids dans M5 (orange : plus d'échecs · bleu : moins d'échecs)",
                 tickformat="+.2f")  # fmt: skip
ui.show(ui.layout(fig, height=560))
ui.takeaway("Réinstallation de populations, nombreux cofinanceurs, deuxième phase, catégorie "
            "environnementale A et agence à capacité faible vont avec plus d'échecs. Les poids "
            "sont petits (forte régularisation) : ce sont des tendances, pas des causes.")

st.subheader("Ce que les agents apportent, et ce qu'ils n'apportent pas")
st.markdown(section_markdown(10, m).split("**En technique.**", 1)[1])
