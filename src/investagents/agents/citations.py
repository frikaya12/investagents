"""Vérification des citations (règle n° 4) : chaque citation doit exister dans le document.

Comparaison sur le texte normalisé (espaces compactés, minuscules, guillemets et tirets
unifiés). Une citation est « retrouvée » si elle est une sous-chaîne exacte du document après
cette normalisation. Pas de correspondance approximative : un LLM qui paraphrase échoue.
"""

from __future__ import annotations

import re

_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _norm(s: str) -> str:
    s = s.translate(_TRANSLATE).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .\"'")


def quote_found(quote: str, source: str) -> bool:
    q = _norm(quote)
    return len(q) >= 10 and q in _norm(source)


def citation_rate(quotes: list[str], source: str) -> float:
    if not quotes:
        return 0.0
    return sum(quote_found(q, source) for q in quotes) / len(quotes)
