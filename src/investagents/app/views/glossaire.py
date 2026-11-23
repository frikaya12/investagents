"""Glossaire : le même que le rapport, avec une recherche."""

import unicodedata

import streamlit as st

from investagents.app import data, ui
from investagents.report.build import fill
from investagents.report.glossary import TERMS

m = data.metrics()


def norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


st.title("Glossaire")
ui.intro("Chaque terme technique de la démo et du rapport, en simple, en technique, et avec un "
         "exemple tiré du projet.")
q = st.text_input("Rechercher un terme", placeholder="ex. AUC, calibration, coffre…")
terms = sorted(TERMS, key=lambda x: x.term.lower())
if q:
    terms = [x for x in terms if norm(q) in norm(x.term + " " + x.simple + " " + x.technique)]
st.caption(f"{len(terms)} terme(s).")
for term in terms:
    with st.expander(term.term, expanded=bool(q)):
        st.markdown(f"**En simple.** {fill(term.simple, m)}")
        st.markdown(f"**En technique.** {fill(term.technique, m)}")
        st.markdown(f"**Dans notre projet.** {fill(term.exemple, m)}")
