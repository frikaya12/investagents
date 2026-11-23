"""Éléments d'interface communs à toutes les pages de la démo.

- palette et gabarit Plotly cohérents (mêmes couleurs que le rapport et la vitrine) ;
- `intro()` : la phrase qui dit à quoi sert la page ;
- `takeaway()` : la ligne « Ce qu'il faut retenir » sous chaque graphique ;
- `gloss()` : le texte d'infobulle (help=) d'un terme, tiré du glossaire du rapport ;
- `card()` : petite carte HTML (fond clair, bordure) pour le schéma du pipeline et les arguments.
"""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

from investagents.report import glossary

BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#7a7974", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e1", "#fcfcfb"
BLUE_SOFT, ORANGE_SOFT = "#e8f0fb", "#fdeee7"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1 {{ letter-spacing: -0.01em; }}
.ia-intro {{ font-size: 1.15rem; color: {INK2}; margin: -0.4rem 0 1.4rem; max-width: 60rem; }}
.ia-take {{ border-left: 4px solid {BLUE}; background: {BLUE_SOFT}; padding: .65rem 1rem;
  border-radius: 0 10px 10px 0; margin: .2rem 0 1.6rem; color: {INK}; }}
.ia-take b {{ color: {BLUE}; }}
.ia-card {{ border: 1px solid #e2e1dc; border-radius: 12px; padding: .9rem 1rem; background: #fff;
  height: 100%; }}
.ia-card h4 {{ margin: 0 0 .3rem; font-size: 1rem; }}
.ia-card p {{ margin: 0; color: {INK2}; font-size: .92rem; }}
.ia-flow {{ display: flex; align-items: stretch; gap: .5rem; flex-wrap: wrap;
  margin: .4rem 0 1rem; }}
.ia-flow .ia-card {{ flex: 1 1 150px; border-top: 4px solid {BLUE}; }}
.ia-arrow {{ align-self: center; color: {GREY}; font-size: 1.4rem; }}
.ia-quote {{ border-left: 3px solid #d6d4cc; padding: .3rem .7rem; margin: .35rem 0;
  font-size: .9rem; color: {INK2}; background: #faf9f6; border-radius: 0 6px 6px 0; }}
.ia-quote mark {{ background: #fff1b8; padding: 0 .15rem; border-radius: 3px; color: {INK}; }}
.ia-badge {{ display: inline-block; font-size: .75rem; font-weight: 600; padding: .05rem .5rem;
  border-radius: 999px; margin-right: .3rem; }}
.ia-pro {{ background: {BLUE_SOFT}; color: #1c5cab; }}
.ia-con {{ background: {ORANGE_SOFT}; color: #a8431b; }}
.ia-neutral {{ background: #eeede8; color: {INK2}; }}
.ia-message {{ font-size: 1.12rem; font-style: italic; border: 1px solid #e2e1dc;
  border-radius: 12px; padding: 1rem 1.2rem; background: #fff; }}
</style>
"""


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def intro(text: str) -> None:
    st.markdown(f'<p class="ia-intro">{text}</p>', unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f'<div class="ia-take"><b>Ce qu\'il faut retenir :</b> {text}</div>',
                unsafe_allow_html=True)  # fmt: skip


def gloss(term: str) -> str:
    """Infobulle d'un terme technique : définition simple + renvoi à la page Glossaire."""
    t = glossary.find(term)
    return f"**{t.term}** : {t.simple} (Voir la page Glossaire.)"


def card(title: str, body: str, extra_class: str = "") -> str:
    return (f'<div class="ia-card {extra_class}"><h4>{escape(title)}</h4>'
            f"<p>{escape(body)}</p></div>")  # fmt: skip


def badge(text: str, kind: str) -> str:
    return f'<span class="ia-badge ia-{kind}">{escape(text)}</span>'


def quote(text: str) -> str:
    return f'<div class="ia-quote">« <mark>{escape(text)}</mark> »</div>'


def layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    """Gabarit Plotly commun : fond clair, grille discrète, police système, marges serrées."""
    base = dict(
        height=height, template="none", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "size": 13,
              "color": INK},
        margin={"l": 10, "r": 20, "t": 30, "b": 40}, hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": INK2}},
    )  # fmt: skip
    fig.update_layout(**{**base, **kw})  # les réglages de la page priment
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": INK2})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": INK2})
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
