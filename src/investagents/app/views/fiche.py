"""Fiche projet : un vrai projet de validation, vu par le ML et par chaque agent.

L'issue IEG réelle n'est montrée qu'après le bouton « Révéler l'issue ».
"""

import random

import plotly.graph_objects as go
import streamlit as st

from investagents.app import data, ui
from investagents.app.project_sheet import sheet
from investagents.report.metrics import CRITERIA_FR, FACTOR_FR, pct

FACTOR_LABELS = {**FACTOR_FR, "env_category": "Catégorie environnementale",
                 "first_operation": "Première opération (exclu des modèles)"}  # fmt: skip
VALUE_FR = {"yes": "oui", "no": "non", "high": "élevé", "substantial": "substantiel",
            "moderate": "modéré", "low": "faible", "modest": "modeste"}  # fmt: skip
VERDICT = {"strong": ("argument fort", "pro"), "weak": ("argument faible", "neutral"),
           "unsupported": ("non étayé", "con")}  # fmt: skip

recs, meta = data.committee()
valid = meta[meta.split == "valid"].sort_values("key").reset_index(drop=True)
factors = data.factors_by_project()
cites = data.factor_citations()

st.title("Fiche projet")
ui.intro("Un vrai projet de la Banque mondiale (validation 2008–2011), vu par le modèle de machine "
         "learning puis par chaque agent. L'issue réelle reste cachée jusqu'à ce que vous la "
         "révéliez.")  # fmt: skip


def label(k: str) -> str:
    row = valid.set_index("key").loc[k]
    return f"{k} · {row.country} · {row.global_practice} · {row.approval_fy}"


if "fiche_key" not in st.session_state:
    st.session_state.fiche_key = valid.key.iloc[0]
c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
key = c1.selectbox("Projet (identifiant masqué · pays · secteur · année d'approbation)",
                   valid.key.tolist(), format_func=label, key="fiche_key")  # fmt: skip


def _random_project() -> None:
    st.session_state.fiche_key = random.choice(valid.key.tolist())


c2.button("🎲 Un projet au hasard", width="stretch", on_click=_random_project)

row = valid[valid.key == key].iloc[0]
s = sheet(recs[key], row, factors_lookup=factors.get)
scores = {x["criterion"]: x for x in s["scores"]}

amount = f"{s['amount_usd'] / 1e6:,.0f} M USD".replace(",", " ") if s["amount_usd"] else "—"
st.caption(f"Pays : **{s['country']}** · Secteur : **{s['sector']}** · Montant à l'approbation : "
           f"**{amount}** · Ce que les agents ont lu : le dossier d'évaluation, noms masqués.")

# --- 1. ML : jauge ; 2. Juge : radar ---
g1, g2 = st.columns(2)
with g1:
    st.subheader("Ce que dit le machine learning")
    mean_p = float(valid.p_success.mean())
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * s["p_success"],
        number={"suffix": " %", "valueformat": ".0f", "font": {"size": 46}},
        title={"text": "Probabilité de réussite (M1-texte calibré)", "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": " %"}, "bar": {"color": ui.BLUE},
               "bgcolor": "#f1f0ec", "borderwidth": 0,
               "threshold": {"line": {"color": ui.INK, "width": 3}, "thickness": 0.8,
                             "value": 100 * mean_p}},
    ))  # fmt: skip
    ui.show(ui.layout(fig, height=290, margin={"l": 30, "r": 30, "t": 60, "b": 10}))
    ui.takeaway(f"Le trait noir est la moyenne des projets de validation ({pct(mean_p, 0)}). Le "
                "modèle classe les projets un peu mieux que le hasard : c'est une tendance, pas "
                "une certitude sur ce projet.")  # fmt: skip
with g2:
    st.subheader("Ce que dit le Juge (5 critères)")
    labels = [CRITERIA_FR[c] for c in data.CRITERIA]
    vals = [scores[c]["score"] for c in data.CRITERIA]
    pool = data.decision_pool()
    avg = [float(pool[c].mean()) for c in data.CRITERIA]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=avg + avg[:1], theta=labels + labels[:1], name="moyenne",
                                  line={"color": ui.GREY, "dash": "dot"}))  # fmt: skip
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], name="ce projet",
                                  fill="toself", line={"color": ui.BLUE},
                                  fillcolor="rgba(42,120,214,0.18)"))  # fmt: skip
    fig.update_layout(polar={"radialaxis": {"range": [0, 10], "tickvals": [0, 5, 10],
                                            "tickfont": {"size": 10}},
                             "bgcolor": ui.SURFACE})  # fmt: skip
    ui.show(ui.layout(fig, height=290, margin={"l": 60, "r": 60, "t": 40, "b": 20}))
    ui.takeaway("Les notes du Juge (sur 10) résument le débat des agents. Elles servent à "
                "expliquer "
                "et à classer selon vos priorités ; elles ne prédisent pas l'échec.")

# --- 3. Le débat, critère par critère ---
st.subheader("Le débat des agents, critère par critère")
st.caption("Pour chaque critère, un avocat Pour et un avocat Contre argumentent en citant le "
           "dossier ; un critique juge chaque argument ; le Juge tranche. Textes rédigés en "
           "anglais "
           "par le LLM (Qwen2.5 7B via Ollama), comme les dossiers ; phrases citées surlignées, "
           "toutes vérifiées en Python.")  # fmt: skip
tabs = st.tabs([f"{CRITERIA_FR[c]} · {scores[c]['score']}/10" for c in data.CRITERIA])
for tab, c in zip(tabs, data.CRITERIA, strict=True):
    with tab:
        st.markdown(f"**Le Juge ({scores[c]['score']}/10) :** {scores[c]['justification']}")
        pair = [x for x in s["arguments"] if x["criterion"] == c]
        if len(pair) == 2 and pair[0]["cited"] == pair[1]["cited"]:
            st.info(f"Les deux avocats citent ici les mêmes phrases (cas de "
                    f"{pct(data.identical_citation_share(), 0)} des paires) : une citation exacte "
                    "ne garantit pas qu'un argument soit pertinent.", icon="ℹ️")
        cols = st.columns(2)
        for col, stance, title, kind in ((cols[0], "pro", "Pour", "pro"),
                                         (cols[1], "contra", "Contre", "con")):  # fmt: skip
            for a in [x for x in s["arguments"] if x["criterion"] == c and x["stance"] == stance]:
                v_label, v_kind = VERDICT.get(a["critic_verdict"], ("—", "neutral"))
                html = (ui.badge(title, kind) + ui.badge(f"critique : {v_label}", v_kind)
                        + f"<p style='margin:.5rem 0'>{ui.escape(a['claim'])}</p>"
                        + "".join(ui.quote(q[:400]) for q in a["cited"][:3]))  # fmt: skip
                if not a["citation_valid"]:
                    html += ui.badge("citation non vérifiée", "neutral")
                col.markdown(f'<div class="ia-card">{html}</div>', unsafe_allow_html=True)

# --- 4. Facteurs de risque extraits ---
st.subheader("Les facteurs de risque relevés par l'agent extracteur")
f = s["factors"] or {}
sent = cites.get(row.project_id, {}).get("sentences", {})
found = {k: v for k, v in f.items() if v.get("status") == "found" and v.get("consistent")
         and v.get("ids_valid", True)}  # fmt: skip
if not f:
    st.info("Pas de facteurs extraits pour ce projet.")
else:
    st.caption(f"{len(found)} facteurs sur {len(f)} trouvés dans le dossier, chacun avec sa "
               "phrase source ; « non trouvé » est une réponse autorisée.")  # fmt: skip
    cols = st.columns(2)
    ordered = sorted(found.items(), key=lambda kv: FACTOR_LABELS.get(kv[0], kv[0]))
    for i, (k, v) in enumerate(ordered):
        value = v.get("value")
        value = VALUE_FR.get(str(value).lower(), value)
        quotes = [sent[str(j)] for j in v.get("sentence_ids", [])[:2] if str(j) in sent]
        html = (f"<h4>{ui.escape(FACTOR_LABELS.get(k, k))} : {ui.escape(str(value))}</h4>"
                + "".join(ui.quote(q) for q in quotes[:1]))  # fmt: skip
        cols[i % 2].markdown(f'<div class="ia-card" style="margin-bottom:.6rem">{html}</div>',
                             unsafe_allow_html=True)  # fmt: skip
    missing = sorted(FACTOR_LABELS.get(k, k) for k, v in f.items() if k not in found)
    if missing:
        st.caption("Non trouvés : " + ", ".join(missing) + ".")
ui.takeaway("Ces facteurs (réinstallation, cofinanceurs, capacité de l'agence…) portent un vrai "
            "signal, faible, et rendent le risque lisible ; ils n'ajoutent rien au texte complet "
            "pour prédire.")  # fmt: skip

# --- 5. L'issue réelle, à la demande ---
st.subheader("Et en vrai ?")
state_key = f"reveal_{key}"
if st.button("Révéler l'issue IEG", type="primary", key=f"btn_{key}"):
    st.session_state[state_key] = True
if st.session_state.get(state_key):
    fail = bool(row.y_fail)
    msg = (f"**{row.project_name}** · note IEG : **{row.outcome}** → "
           f"{'échec (MU/U/HU)' if fail else 'succès (HS/S/MS)'}. Le ML donnait "
           f"{pct(s['p_success'], 0)} de chances de réussite.")  # fmt: skip
    (st.error if fail else st.success)(msg)
    st.caption("Cette note a été attribuée après la fin du projet ; ni le ML ni les agents ne "
               "l'ont vue. Un seul projet ne dit rien de la qualité du modèle : voir la page "
               "Résultats.")  # fmt: skip
else:
    st.caption("Le nom du projet et sa note IEG restent cachés, comme pour les agents.")
