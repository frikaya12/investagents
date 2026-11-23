"""Décision multicritère : curseur α + poids des 5 critères + budget -> portefeuille en direct.

Validation 2008–2011 seulement (choisir α en regardant le test reviendrait à régler sur le test).
Même code que la frontière α (selection/frontier.py) : à poids égaux et budget 25 %, les points
α = 1, 0,75, 0,5, 0,25, 0 retombent exactement sur ceux de experiments/blocC/frontier_valid.json.
"""

import plotly.graph_objects as go
import streamlit as st

from investagents.app import data, ui
from investagents.report.metrics import CRITERIA_FR, pct

st.title("Décision multicritère")
ui.intro("Vous êtes le décideur : réglez l'équilibre entre les chances de réussite (machine "
         "learning) et le jugement du comité d'agents, choisissez vos priorités et votre budget. "
         "Le portefeuille se recalcule en direct.")  # fmt: skip
st.warning("Les notes du comité servent à classer selon **vos** priorités : elles **ne prédisent "
           "pas l'échec**. Projets de validation (2008–2011) seulement ; aucun α n'est recommandé "
           "à partir du test.", icon="⚠️")  # fmt: skip

c1, c2 = st.columns([3, 2])
alpha = c1.slider("α : poids de la probabilité de réussite (1 = ML seul, 0 = comité seul)",
                  0.0, 1.0, 0.5, 0.05, help=ui.gloss("frontière α"))  # fmt: skip
budget = c2.select_slider("Budget (part du montant de chaque cohorte annuelle)",
                          options=[0.10, 0.25, 0.50], value=0.25,
                          format_func=lambda f: f"{f:.0%}")  # fmt: skip
st.markdown("**Vos priorités** (poids des 5 critères du Juge, de 0 à 5)")
cols = st.columns(5)
weights = tuple(cols[i].slider(CRITERIA_FR[c], 0, 5, 1, key=f"w_{c}")
                for i, c in enumerate(data.CRITERIA))  # fmt: skip
if sum(weights) == 0:
    st.error("Mettez au moins un poids non nul.")
    st.stop()

pool = data.decision_pool()
chosen = data.select_portfolio(alpha, weights, budget)
impact = float(chosen.impact_sustainability.mean())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Projets financés", f"{len(chosen)} / {len(pool)}", border=True)
k2.metric("Réussite attendue (ML)", pct(float(chosen.p_success.mean())),
          f"{100 * (chosen.p_success.mean() - pool.p_success.mean()):+.1f} pts vs tous"
          .replace(".", ","),
          help="Moyenne des probabilités de réussite du ML pour les projets choisis.",
          border=True)  # fmt: skip
k3.metric("Note d'impact moyenne (Juge)", f"{impact:.2f} / 10".replace(".", ","),
          f"{impact - pool.impact_sustainability.mean():+.2f} vs tous".replace(".", ","),
          border=True)  # fmt: skip
k4.metric("Réussis en réalité (IEG)", pct(float(chosen.success.mean())),
          help="Connu a posteriori, sur la validation : sert à vérifier, pas à choisir.",
          border=True)  # fmt: skip

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Carte de la frontière α** (validation, budget 25 %, poids égaux)")
    fr = data.frontier("valid")
    pts = fr["frontier"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["mean_impact_sustainability"] for p in pts],
        y=[p["count_share_satisfactory"] for p in pts], mode="lines+markers+text",
        text=[f"α={p['alpha']:g}".replace(".", ",") for p in pts],
        textposition=["top right"] * (len(pts) - 1) + ["bottom left"],
        name="frontière (α fixés à l'avance)", line={"color": ui.GREY}, marker={"size": 8},
        hovertemplate="impact %{x:.2f} · réussite %{y:.1%}<extra></extra>",
    ))  # fmt: skip
    rnd = fr["random"]["mean"]
    fig.add_trace(go.Scatter(x=[rnd["mean_impact_sustainability"]],
                             y=[rnd["count_share_satisfactory"]], mode="markers", name="hasard",
                             marker={"symbol": "x", "size": 12, "color": ui.GREY}))  # fmt: skip
    fig.add_trace(go.Scatter(x=[impact], y=[float(chosen.success.mean())], mode="markers",
                             name="votre portefeuille",
                             marker={"size": 18, "color": ui.BLUE, "symbol": "star",
                                     "line": {"width": 1, "color": "white"}}))  # fmt: skip
    fig.update_xaxes(title="Note d'impact moyenne du portefeuille (sur 10)")
    ys = [p["count_share_satisfactory"] for p in pts] + [float(chosen.success.mean()),
                                                         rnd["count_share_satisfactory"]]
    fig.update_yaxes(title="Part de projets réussis (IEG)", tickformat=".0%",
                     range=[min(ys) - 0.03, max(ys) + 0.03])
    ui.show(ui.layout(fig, height=380))
    ui.takeaway("Vers la droite (α petit), le portefeuille suit le comité et gagne en note "
                "d'impact ; vers le haut (α grand), il suit le ML. Sur la validation, ce gain "
                "d'impact se paie en projets réussis.")  # fmt: skip
with g2:
    st.markdown("**Profil du portefeuille sur les 5 critères**")
    labels = [CRITERIA_FR[c] for c in data.CRITERIA]
    a = [float(chosen[c].mean()) for c in data.CRITERIA]
    b = [float(pool[c].mean()) for c in data.CRITERIA]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=b + b[:1], theta=labels + labels[:1], name="tous les projets",
                                  line={"color": ui.GREY, "dash": "dot"}))  # fmt: skip
    fig.add_trace(go.Scatterpolar(r=a + a[:1], theta=labels + labels[:1], name="portefeuille",
                                  fill="toself", line={"color": ui.BLUE},
                                  fillcolor="rgba(42,120,214,0.18)"))  # fmt: skip
    fig.update_layout(polar={"radialaxis": {"range": [5, 8.5]}, "bgcolor": ui.SURFACE})
    ui.show(ui.layout(fig, height=380, margin={"l": 60, "r": 60, "t": 40, "b": 20}))
    ui.takeaway("Les écarts sont petits (axe de 5 à 8,5) : le Juge note serré. Augmenter le poids "
                "d'un critère pousse le portefeuille vers les projets bien notés sur ce critère.")

with st.expander("Voir les projets choisis"):
    show = chosen[["key", "country", "global_practice", "cost", "p_success", *data.CRITERIA]]
    st.dataframe(show.rename(columns={"key": "projet", "country": "pays",
                                      "global_practice": "secteur", "cost": "montant (USD)",
                                      "p_success": "p réussite (ML)",
                                      **{c: CRITERIA_FR[c] for c in data.CRITERIA}}),
                 hide_index=True, width="stretch")  # fmt: skip
st.caption("score = α × p_réussite + (1 − α) × moyenne pondérée des notes / 10 ; sac à dos "
           "OR-Tools "
           "CP-SAT (objectif figé Σ score × montant), une sélection par cohorte annuelle, au plus "
           "25 % du budget par pays.")  # fmt: skip
