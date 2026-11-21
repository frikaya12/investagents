"""Masquage des noms de pays, de lieux et de nationalités (bloc B, variante « texte masqué »).

But : mesurer l'information du texte AU-DELÀ de l'identité du pays (en phase 2, les mots les plus
influents de M1-texte étaient souvent des noms de pays).

Trois passes :
1. reconnaissance d'entités spaCy (`en_core_web_sm` 3.8.0) : entités GPE (pays, villes, régions),
   LOC (lieux géographiques) et NORP (nationalités : « Chadian ») remplacées par [PLACE] ;
2. liste de pays (relevés de prêts IDA/IBRD + GeoNames + anciens noms), sans tenir compte de la
   casse ni des accents ;
3. liste des capitales (GeoNames, via le paquet `geonamescache`, données CC BY 4.0), sensible à la
   casse (nom propre) mais pas aux accents : « Lome » et « Lomé » sont masqués.
Limite connue : les autres villes, les régions et les fautes d'OCR ne sont rattrapées que par
spaCy ; le masquage réduit l'information pays, il ne la supprime pas totalement
(mesure : reports/masquage_lieux.md).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

PLACE_LABELS = {"GPE", "LOC", "NORP"}
TOKEN = "[PLACE]"

# Anciens noms et variantes courantes dans les dossiers 1970–2017 (liste fixe, écrite à la main).
HISTORICAL_COUNTRIES = [
    "Zaire", "Upper Volta", "Dahomey", "Burma", "Ceylon", "Rhodesia", "Swaziland",
    "Kampuchea", "Czechoslovakia", "Yugoslavia", "Soviet Union", "USSR", "Ivory Coast",
    "Cote d'Ivoire", "Viet Nam", "Lao PDR", "Laos", "East Timor", "Timor-Leste",
    "Yemen Arab Republic", "Cape Verde", "Cabo Verde", "Macedonia", "Kyrgyz Republic",
    "Slovak Republic", "Gambia", "Bahamas",
]  # fmt: skip
# Capitales qui sont aussi des mots anglais courants : exclues pour ne pas masquer « male ».
AMBIGUOUS_CAPITALS = {"Male", "The Valley", "Road Town", "Stanley", "Hamilton", "Victoria"}

_ACCENTS = {
    "a": "aàáâãäå", "c": "cç", "e": "eèéêë", "i": "iìíîï", "n": "nñ",
    "o": "oòóôõöø", "u": "uùúûü", "y": "yýÿ",
}  # fmt: skip


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _accent_insensitive(name: str) -> str:
    """Motif regex d'un nom où chaque voyelle accepte ses variantes accentuées."""
    out = []
    for ch in strip_accents(name):
        low = ch.lower()
        if low in _ACCENTS:
            chars = _ACCENTS[low].upper() if ch.isupper() else _ACCENTS[low]
            out.append(f"[{chars}]")
        elif ch in " -":
            out.append(r"[\s\-]+")
        elif ch == "'":
            out.append(r"['’]?")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _pattern(names, ignore_case: bool) -> re.Pattern | None:
    names = sorted(
        {strip_accents(n.strip()) for n in names if isinstance(n, str) and len(n.strip()) > 3},
        key=len,
        reverse=True,
    )
    if not names:
        return None
    body = "|".join(_accent_insensitive(n) for n in names)
    return re.compile(r"\b(?:" + body + r")\b", re.IGNORECASE if ignore_case else 0)


def country_pattern(names) -> re.Pattern | None:
    """Noms de pays, sans casse ni accents (« Congo, Democratic Republic of » → aussi « Congo »)."""
    expanded = set()
    for n in names:
        if isinstance(n, str):
            expanded.add(n)
            expanded.add(n.split(",")[0])
    return _pattern(expanded, ignore_case=True)


@lru_cache(maxsize=1)
def gazetteer() -> tuple[list[str], list[str]]:
    """(pays, capitales) de la liste fixe GeoNames + anciens noms."""
    import geonamescache

    countries = geonamescache.GeonamesCache().get_countries().values()
    names = [c["name"] for c in countries] + HISTORICAL_COUNTRIES
    capitals = [c["capital"] for c in countries if c["capital"]]
    capitals = [c for c in capitals if c not in AMBIGUOUS_CAPITALS]
    return names, capitals


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load(
        "en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"]
    )


def spacy_mask(texts: list[str], batch_size: int = 16) -> list[str]:
    """Passe 1 : entités GPE / LOC / NORP remplacées par [PLACE]."""
    out = []
    for doc in _nlp().pipe(texts, batch_size=batch_size):
        pieces, last = [], 0
        for ent in doc.ents:
            if ent.label_ in PLACE_LABELS:
                pieces.append(doc.text[last : ent.start_char])
                pieces.append(TOKEN)
                last = ent.end_char
        pieces.append(doc.text[last:])
        out.append("".join(pieces))
    return out


def list_mask(texts: list[str], country_names=(), use_gazetteer: bool = True) -> list[str]:
    """Passes 2 et 3 : listes fixes de pays et de capitales."""
    countries, capitals = gazetteer() if use_gazetteer else ([], [])
    pats = [country_pattern(list(country_names) + countries), _pattern(capitals, False)]
    pats = [p for p in pats if p is not None]
    out = []
    for text in texts:
        for p in pats:
            text = p.sub(TOKEN, text)
        out.append(text)
    return out


def mask_places(
    texts: list[str], country_names=(), batch_size: int = 16, use_gazetteer: bool = True
) -> list[str]:
    """Renvoie les textes avec pays, capitales, lieux et nationalités remplacés par [PLACE]."""
    return list_mask(spacy_mask(texts, batch_size), country_names, use_gazetteer)
