"""Résultats : AUC-PR, coffre-fort, écarts appariés, calibration, backtest.

Tous les chiffres viennent de report.metrics (les JSON de résultats). Aucun calcul de modèle.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from investagents.app import data, ui
from investagents.ml.calibration import ece
from investagents.report.metrics import chart_auc, chart_diffs, integer, num, pct

m = data.metrics()
t, r = m.txt, m.raw

st.title("Résultats")
ui.intro("Tous les modèles mesurés une seule fois sur des projets qu'ils n'avaient jamais "
         "vus, avec "
         "leur marge d'incertitude. Les résultats négatifs sont montrés comme les positifs.")

# --- 1. AUC-PR des modèles ---
st.subheader("Qui prédit l'échec ? AUC-PR de chaque modèle")
rows = chart_auc(m)
fig = go.Figure()
names = [f"{x['short']} · {x['desc']}" for x in rows]
colors = [ui.ORANGE if "coffre" in x["short"] else ui.BLUE for x in rows]
fig.add_trace(go.Scatter(
    x=[x["base"] for x in rows], y=names, mode="markers", name="taux de base (hasard)",
    marker={"symbol": "line-ns", "size": 18, "line": {"width": 3, "color": ui.GREY}},
    hovertemplate="hasard : %{x:.3f}<extra></extra>",
))  # fmt: skip
fig.add_trace(go.Scatter(
    x=[x["value"] for x in rows], y=names, mode="markers", name="AUC-PR (IC 95 %)",
    marker={"size": 12, "color": colors, "line": {"width": 2, "color": "white"}},
    error_x={"type": "data", "symmetric": False, "array": [x["hi"] - x["value"] for x in rows],
             "arrayminus": [x["value"] - x["lo"] for x in rows], "color": ui.BLUE, "thickness": 2},
    customdata=[[x["lo"], x["hi"], x["n"], x["sample"]] for x in rows],
    hovertemplate="%{y}<br>AUC-PR %{x:.3f} [%{customdata[0]:.3f} ; %{customdata[1]:.3f}]"
                  "<br>%{customdata[3]}, n = %{customdata[2]}<extra></extra>",
))  # fmt: skip
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="AUC-PR (classe « échec »)", range=[0.1, 0.45], tickformat=".2f")
ui.show(ui.layout(fig, height=430))
ui.takeaway(f"Tous les modèles restent proches de leur taux de base. Seul le texte du dossier "
            f"(M1-texte, {num(r['m1t_aucpr']['value'])}, et M6 qui le contient) s'en écarte "
            "nettement. En orange, le coffre-fort : M1 y retombe au niveau du hasard.")

# --- 2. Coffre-fort ---
st.subheader("Le coffre-fort : des projets neufs, ouverts une seule fois à la fin")
c1, c2, c3 = st.columns(3)
c1.metric("M1 · AUC-ROC sur le coffre-fort", num(r["vault_m1_aucroc"]["value"]),
          help=ui.gloss("AUC-ROC"), border=True)  # fmt: skip
c1.caption(f"IC {t['vault_m1_aucroc'].split(' ', 1)[1]} : contient 0,5, le hasard.")
c2.metric("M1 · AUC-PR sur le coffre-fort", num(r["vault_m1_aucpr"]["value"]),
          help=ui.gloss("coffre-fort"), border=True)  # fmt: skip
c2.caption(f"contre un taux de base de {t['vault_m0_aucpr']} ({t['vault_n']} projets).")
c3.metric("M1-texte − M1 sur le coffre-fort", num(r["vault_m1t_minus_m1"]["value"], signed=True),
          help=ui.gloss("écart apparié"), border=True)  # fmt: skip
c3.caption(f"sur {t['vault_text_n']} projets seulement ({t['vault_text_fail_n']} échecs) : "
           "descriptif, non concluant.")  # fmt: skip
ui.takeaway("Sur des projets plus récents que tout l'apprentissage, le modèle à variables "
            "chiffrées "
            "ne prédit plus rien. Les chiffres du test 2012–2017 étaient donc optimistes : le "
            "signal est faible et fragile dans le temps.")  # fmt: skip

# --- 2 bis. AUC-ROC (0,5 = hasard pour tous) ---
st.subheader("L'autre mesure : AUC-ROC de chaque modèle (0,5 = hasard)")
ROC_ROWS = [
    ("M1 · test complet", "roc_m1"), ("M1 · projets avec texte", "roc_m1_text_subset"),
    ("M1-texte", "roc_m1t"), ("Texte seul · test du comité", "roc_c_texte"),
    ("M3 · notes du Juge", "roc_m3"), ("M4 · texte + notes", "roc_m4"),
    ("M7 · texte + débat", "roc_m7"), ("Texte seul · test du bloc D", "roc_d_texte"),
    ("M5 · 19 facteurs", "roc_m5"), ("M6 · texte + facteurs", "roc_m6"),
    ("M1 · coffre-fort", "roc_vault_m1"),
]  # fmt: skip
roc = [(name, r[k]) for name, k in ROC_ROWS]
above = [d["ci_low"] > 0.5 for _, d in roc]
fig = go.Figure(go.Scatter(
    x=[d["value"] for _, d in roc], y=[n for n, _ in roc], mode="markers", showlegend=False,
    marker={"size": 12, "color": [ui.ORANGE if "coffre" in n else (ui.BLUE if a else "white")
                                  for (n, _), a in zip(roc, above, strict=True)],
            "line": {"width": 2, "color": [ui.BLUE if a else ui.GREY for a in above]}},
    error_x={"type": "data", "symmetric": False,
             "array": [d["ci_high"] - d["value"] for _, d in roc],
             "arrayminus": [d["value"] - d["ci_low"] for _, d in roc], "color": ui.GREY,
             "thickness": 2},
    customdata=[[d["ci_low"], d["ci_high"]] for _, d in roc],
    hovertemplate="%{y}<br>AUC-ROC %{x:.3f} [%{customdata[0]:.3f} ; %{customdata[1]:.3f}]"
                  "<extra></extra>",
))  # fmt: skip
fig.add_vline(x=0.5, line={"color": ui.INK2, "width": 1.5},
              annotation={"text": "hasard (0,5)", "font": {"color": ui.INK2}})  # fmt: skip
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="AUC-ROC (IC 95 %) — point plein : intervalle au-dessus de 0,5",
                 range=[0.35, 0.75], tickformat=".2f")  # fmt: skip
ui.show(ui.layout(fig, height=430))
yes = ", ".join(n for (n, _), a in zip(roc, above, strict=True) if a)
no = ", ".join(n for (n, _), a in zip(roc, above, strict=True) if not a)
ui.takeaway(f"Au mieux {t['roc_max']} : le classement reste modeste. Intervalle au-dessus du "
            f"hasard : {yes}. Pas distinguable du hasard : {no}. Chaque modèle est mesuré sur son "
            "propre échantillon : pour comparer deux modèles, voir les écarts appariés plus bas.")
with st.expander("Voir le tableau des AUC-ROC"):
    st.dataframe(pd.DataFrame([{"modèle": n, "AUC-ROC": round(d["value"], 3),
                                "IC bas": round(d["ci_low"], 3),
                                "IC haut": round(d["ci_high"], 3),
                                "IC au-dessus de 0,5": "oui" if a else "non"}
                               for (n, d), a in zip(roc, above, strict=True)]),
                 hide_index=True, width="stretch")  # fmt: skip

# --- 2 ter. Bloc E : sans les noms de pays ---
st.subheader("Le texte prédit-il seulement parce qu'il nomme le pays ?")
c1, c2, c3 = st.columns(3)
c1.metric("M1-texte d'origine · AUC-PR", num(r["e_orig_aucpr"]["value"]),
          help=ui.gloss("AUC-PR"), border=True)  # fmt: skip
c1.caption(f"IC {t['e_orig_aucpr'].split(' ', 1)[1]}")
c2.metric("Sans noms de pays ni adjectifs · AUC-PR", num(r["e_nc_aucpr"]["value"]),
          help=ui.gloss("AUC-PR"), border=True)  # fmt: skip
c2.caption(f"IC {t['e_nc_aucpr'].split(' ', 1)[1]} · même modèle, réentraîné sans ces mots")
c3.metric("Écart apparié", num(r["e_diff_aucpr"]["value"], signed=True),
          help=ui.gloss("écart apparié"), border=True)  # fmt: skip
c3.caption(f"IC {t['e_diff_aucpr'].split(' ', 1)[1]} · {t['e_share_lost']} du gain perdu")
ui.takeaway(f"Sans noms de pays : {num(r['e_nc_aucpr']['value'])} contre "
            f"{num(r['e_orig_aucpr']['value'])}. Le signal vient {t['e_verdict']} (règle fixée "
            "avant le calcul). Nuance : capitales, régions et institutions restent dans le texte ; "
            "le modèle ne dépend pas des noms de pays, mais on ne peut pas dire qu'il ignore le "
            "pays.")

# --- 3. Écarts appariés ---
st.subheader("Qu'est-ce qui apporte vraiment quelque chose ? Écarts appariés")
d = chart_diffs(m)
sure = [x["lo"] > 0 for x in d]
fig = go.Figure(go.Scatter(
    x=[x["value"] for x in d], y=[f"{x['label']} · {x['question']}" for x in d], mode="markers",
    marker={"size": 12, "color": [ui.BLUE if s_ else "white" for s_ in sure],
            "line": {"width": 2, "color": [ui.BLUE if s_ else ui.GREY for s_ in sure]}},
    error_x={"type": "data", "symmetric": False, "array": [x["hi"] - x["value"] for x in d],
             "arrayminus": [x["value"] - x["lo"] for x in d], "color": ui.GREY, "thickness": 2},
    customdata=[[x["lo"], x["hi"]] for x in d], showlegend=False,
    hovertemplate="%{y}<br>écart %{x:+.3f} [%{customdata[0]:.3f} ; %{customdata[1]:.3f}]"
                  "<extra></extra>",
))  # fmt: skip
fig.add_vline(x=0, line={"color": ui.INK2, "width": 1.5})
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="Écart d'AUC-PR (IC 95 %) — point plein : intervalle au-dessus de 0",
                 tickformat="+.2f")  # fmt: skip
ui.show(ui.layout(fig, height=420))
ui.takeaway(f"Trois apports sont démontrés (point plein) : M1 sur le hasard, le texte sur M1 "
            f"({t['m1t_minus_m1']}), les facteurs extraits sur le hasard ({t['m5_minus_m0']}). "
            "Aucun apport des agents en plus du texte ne l'est : M4, M6 et M7 touchent zéro.")

# --- 4. Calibration ---
st.subheader("Les probabilités sont-elles honnêtes ? Calibration de M1-texte")
cal = data.calibration_curve()
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], mode="lines", name="calibration parfaite",
                         line={"color": ui.GREY, "dash": "dot"}))  # fmt: skip
fig.add_trace(go.Scatter(
    x=cal.predit, y=cal.observe, mode="lines+markers", name="M1-texte (test, recalibré)",
    marker={"size": 10, "color": ui.BLUE}, line={"color": ui.BLUE, "width": 2},
    customdata=cal.n, hovertemplate="prédit %{x:.0%} · observé %{y:.0%} · %{customdata} "
                                    "projets<extra></extra>",
))  # fmt: skip
fig.update_xaxes(title="Probabilité d'échec annoncée", tickformat=".0%", range=[0, 0.6])
fig.update_yaxes(title="Part d'échecs observée", tickformat=".0%", range=[0, 0.6])
c1, c2 = st.columns([3, 2])
with c1:
    ui.show(ui.layout(fig, height=360))
with c2:
    preds = data.test_predictions()
    st.metric("ECE après calibration", t["ece_after"], help=ui.gloss("ECE"), border=True)
    st.caption(f"{t['ece_before']} avant la calibration de Platt (apprise sur la validation "
               f"seulement). Recalculé ici sur les prédictions du test : "
               f"{num(ece(preds.y_fail, preds.p_fail_m1_text))}.")  # fmt: skip
    st.metric("Brier", f"{t['brier_before']} → {t['brier_after']}", help=ui.gloss("Brier"),
              border=True)  # fmt: skip
ui.takeaway("Les points suivent à peu près la diagonale : quand le modèle annonce un risque plus "
            "élevé, on observe plus d'échecs, dans des proportions proches de l'annonce. Ces "
            "probabilités peuvent donc servir à sélectionner sous budget.")

# --- 5. Backtest contre le hasard ---
st.subheader("Le portefeuille choisi bat-il le hasard ?")
rd = data.random_draws()
draws = np.array(rd["count_share"])
fig = go.Figure(go.Histogram(x=draws, nbinsx=30, marker={"color": "#cfd6df"},
                             name="1 000 portefeuilles au hasard",
                             hovertemplate="%{x:.1%} : %{y} tirages<extra></extra>"))  # fmt: skip
ours = r["pf25"]["count_share_satisfactory"]
m0 = r["pf25_m0"]["count_share_satisfactory"]
fig.add_vline(x=ours, line={"color": ui.BLUE, "width": 3},
              annotation={"text": f"notre portefeuille {pct(ours)}", "font": {"color": ui.BLUE}})
fig.add_vline(x=m0, line={"color": ui.GREY, "width": 2, "dash": "dash"},
              annotation={"text": f"contrôle M0 {pct(m0)}", "font": {"color": ui.INK2}},
              annotation_position="top left")  # fmt: skip
fig.update_xaxes(title="Part de projets réussis (25 % du budget, test 2012–2017)",
                 tickformat=".0%")  # fmt: skip
fig.update_yaxes(title="Nombre de tirages")
ui.show(ui.layout(fig, height=360, showlegend=False))
n_better = int((draws >= ours).sum())
ui.takeaway(f"{n_better} des {integer(len(draws))} portefeuilles tirés au hasard font au "
            "moins aussi bien que le nôtre "
            f"({t['pf25_p']}). Le contrôle M0, qui ne regarde que la taille des projets, reste au "
            f"milieu du hasard : le gain vient des probabilités du ML. Attention : le même modèle "
            f"fait {t['bt_text']} sur un pool à peine différent ({t['bt_pool_gap']} projets de "
            "moins) ; ± 5 points d'écart entre deux modèles ne veulent rien dire.")
st.caption(f"Tirages reproduits avec le code et la graine du backtest figé ; moyenne "
           f"{pct(float(draws.mean()))} identique au fichier de résultats ({t['pf25_random']}).")

with st.expander("Voir les chiffres en tableau"):
    st.dataframe(pd.DataFrame([{"modèle": x["model"], "échantillon": x["sample"], "n": x["n"],
                                "AUC-PR": round(x["value"], 3), "IC bas": round(x["lo"], 3),
                                "IC haut": round(x["hi"], 3), "taux de base": round(x["base"], 3)}
                               for x in rows]), hide_index=True, width="stretch")
