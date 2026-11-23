"""Bloc E : liste des noms de pays et des adjectifs de nationalité, construite AVANT tout résultat.

Sources (aucune ne regarde un modèle, un poids ou une issue) :
1. la colonne « pays » des données (experiments/mvp/predictions.parquet) : noms officiels IEG
   (« Federal Republic of Nigeria »). Les valeurs qui ne contiennent aucun nom de pays de la
   liste standard sont des régions (« Africa », « Central Asia ») : elles sont EXCLUES (on
   mesure la part du pays, pas celle de la région) et listées dans le JSON ;
2. la liste standard mledoze/countries (https://github.com/mledoze/countries, licence ODbL,
   base de REST Countries) : nom courant et nom officiel en anglais, adjectifs anglais
   (demonyms, masculin et féminin). Les variantes (altSpellings) ne sont pas utilisées : elles
   mêlent noms en langue locale et mots courts ambigus ;
3. les anciens noms déjà fixés pour le masquage (agents/masking.py, HISTORICAL_COUNTRIES).

Sorties (experiments/blocE/country_terms.json, empreinte SHA-256) :
- `phrases` : expressions supprimées des textes (mot entier, sans casse ni accents, la plus longue
  d'abord ; les expressions de 3 caractères ou moins sont ignorées par le motif, comme au
  masquage, pour ne pas toucher aux sigles) ;
- `country_tokens` / `adjective_tokens` : mots isolés servant à étiqueter le vocabulaire du
  TF-IDF (« pays » / « adjectif de pays »), hors mots génériques (GENERIC, fixé ici).

Lancer : `uv run python -m investagents.ml.country_terms`
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from investagents.agents.masking import HISTORICAL_COUNTRIES, strip_accents

ROOT = Path(__file__).resolve().parents[3]
MLEDOZE = ROOT / "data/raw/mledoze_countries.json"
PRED = ROOT / "experiments/mvp/predictions.parquet"
OUT = ROOT / "experiments/blocE/country_terms.json"

# Mots qui apparaissent dans des noms de pays mais ne désignent pas un pays à eux seuls.
GENERIC = {
    "republic", "democratic", "people", "peoples", "of", "the", "and", "united", "states",
    "state", "kingdom", "south", "north", "east", "west", "central", "new", "islands", "island",
    "saint", "st", "federal", "federation", "federative", "federated", "arab", "islamic",
    "socialist", "plurinational", "union", "great", "outlying", "minor", "territory",
    "territories", "bolivarian", "cooperative", "co", "operative", "principality",
    "commonwealth", "grand", "duchy", "holy", "see", "sea", "coast", "gulf", "eastern",
    "western", "northern", "southern", "da", "de", "del", "la", "el", "y", "e", "et", "do",
    "dos", "das", "virgin", "heard", "mcdonald", "keeling", "christmas", "jan", "mayen",
    "is", "rep", "dem", "pdr", "sar", "special", "administrative", "region", "oriental",
    "hashemite", "sultanate", "emirate", "emirates", "independent", "nation", "nations",
    "abode", "city", "collectivity", "confederation", "country", "department", "group", "isle",
    "lands", "man", "most", "ocean", "peace", "serene", "sovereign", "upper", "san", "sint",
    "bailiwick", "africa", "america", "republica", "channel", "islander",
}  # fmt: skip


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z][a-z]+", strip_accents(s).lower())


def build() -> dict:
    std = json.loads(MLEDOZE.read_text(encoding="utf-8"))
    names, adjectives = set(), set()
    for c in std:
        names |= {c["name"]["common"], c["name"]["official"]}  # anglais ; pas d'altSpellings
        eng = (c.get("demonyms") or {}).get("eng") or {}
        adjectives |= {v for v in eng.values() if v}
    names |= set(HISTORICAL_COUNTRIES)

    # une valeur de la colonne est un pays si elle contient un nom ou un adjectif de la liste
    known = {strip_accents(n).lower() for n in names | adjectives if len(n) > 3}
    column = sorted(pd.read_parquet(PRED, columns=["country"]).country.dropna().unique())
    kept, regions = [], []
    for v in column:
        low = strip_accents(v).lower().replace("st. ", "saint ")
        hit = any(re.search(r"\b" + re.escape(n) + r"\b", low) for n in known)
        (kept if hit else regions).append(v)
    names |= set(kept)

    phrases = sorted({strip_accents(" ".join(p.split())) for p in names | adjectives},
                     key=lambda s: (-len(s), s))  # fmt: skip
    # étiquettes : un mot qui est à lui seul un adjectif (« french ») -> adjectif ; sinon un mot
    # d'un nom de pays (« guinea », « lanka ») -> pays ; sinon un mot d'adjectif composé -> adjectif
    single_adj = {t for a in adjectives if len(_tokens(a)) == 1 for t in _tokens(a)} - GENERIC
    country_tok = {t for n in names for t in _tokens(n) if len(t) >= 3} - GENERIC - single_adj
    adj_tok = ({t for a in adjectives for t in _tokens(a) if len(t) >= 3} | single_adj) - GENERIC
    adj_tok -= country_tok
    out = {
        "sources": {"column": str(PRED.relative_to(ROOT)), "standard": "mledoze/countries (ODbL)",
                    "standard_sha256": hashlib.sha256(MLEDOZE.read_bytes()).hexdigest(),
                    "historical": "agents/masking.py HISTORICAL_COUNTRIES"},
        "column_values_kept": kept,
        "column_values_excluded_as_regions": regions,
        "generic_words_not_labelled": sorted(GENERIC),
        "phrases": phrases,
        "country_tokens": sorted(country_tok),
        "adjective_tokens": sorted(adj_tok),
    }  # fmt: skip
    body = json.dumps({k: v for k, v in out.items() if k != "sha256"}, ensure_ascii=False,
                      sort_keys=True)  # fmt: skip
    out["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return out


def main() -> None:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out['phrases'])} expressions ; {len(out['country_tokens'])} mots « pays » ; "
          f"{len(out['adjective_tokens'])} adjectifs ; régions exclues : "
          f"{out['column_values_excluded_as_regions']}")


if __name__ == "__main__":
    main()
