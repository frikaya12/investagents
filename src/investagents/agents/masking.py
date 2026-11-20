"""Masquage des noms de pays, de lieux et de nationalités (bloc B, variante « texte masqué »).

But : mesurer l'information du texte AU-DELÀ de l'identité du pays (en phase 2, les mots les plus
influents de M1-texte étaient souvent des noms de pays).

Deux passes :
1. reconnaissance d'entités spaCy (`en_core_web_sm` 3.8.0) : entités GPE (pays, villes, régions),
   LOC (lieux géographiques) et NORP (nationalités : « Chadian ») remplacées par [PLACE] ;
2. liste de noms de pays (relevés de prêts IDA/IBRD, colonne `country`) : rattrape ce que le
   modèle d'entités manque, sans tenir compte de la casse.
Limite connue : le modèle d'entités n'est pas parfait (ex. « N Djamena » reconnu comme une
organisation) ; le masquage réduit l'information pays, il ne la supprime pas totalement.
"""

from __future__ import annotations

import re
from functools import lru_cache

PLACE_LABELS = {"GPE", "LOC", "NORP"}
TOKEN = "[PLACE]"


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load(
        "en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"]
    )


def country_pattern(names) -> re.Pattern | None:
    names = sorted(
        {n.strip() for n in names if isinstance(n, str) and len(n.strip()) > 3},
        key=len,
        reverse=True,
    )
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)


def mask_places(texts: list[str], country_names=(), batch_size: int = 16) -> list[str]:
    """Renvoie les textes avec pays, lieux et nationalités remplacés par [PLACE]."""
    pattern = country_pattern(country_names)
    out = []
    for doc in _nlp().pipe(texts, batch_size=batch_size):
        pieces, last = [], 0
        for ent in doc.ents:
            if ent.label_ in PLACE_LABELS:
                pieces.append(doc.text[last : ent.start_char])
                pieces.append(TOKEN)
                last = ent.end_char
        pieces.append(doc.text[last:])
        text = "".join(pieces)
        if pattern is not None:
            text = pattern.sub(TOKEN, text)
        out.append(text)
    return out
