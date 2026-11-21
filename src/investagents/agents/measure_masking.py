"""Mesure des noms de lieux restants après masquage, sur un échantillon (bloc B).

Compare trois versions des sections clés de N projets tirés au hasard dans l'ENTRAÎNEMENT :
- brut : sans masquage ;
- v1 : spaCy + noms de pays des relevés IDA/IBRD (première version) ;
- v2 : v1 + liste fixe GeoNames (pays, capitales, sans accents) + anciens noms.

Deux compteurs (occurrences pour 10 000 caractères) :
- « pays + capitales » : la liste ajoutée en v2 (donc presque nul en v2 par construction) ;
- « autres villes ≥ 100 000 hab. » : villes GeoNames qui ne sont ni capitales ni utilisées par
  le masquage → mesure INDÉPENDANTE de ce que le masquage laisse passer. Sont exclus les noms
  qui apparaissent aussi en minuscules dans l'échantillon (mots courants : « reading », « split »).

Lancer : `uv run python -m investagents.agents.measure_masking`
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from investagents.agents.masking import gazetteer, list_mask, strip_accents
from investagents.data.sections_corpus import OUT_PATH, statement_country_names
from investagents.ml.split import load_modeling_table

REPORT = Path("reports/masquage_lieux.md")
N_SAMPLE = 300
SEED = 5


def word_pattern(names) -> re.Pattern:
    names = sorted({n for n in names if len(n) > 3}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")


def count(pattern: re.Pattern, texts) -> tuple[int, Counter]:
    c = Counter(m.group(0) for t in texts for m in pattern.finditer(strip_accents(t)))
    return sum(c.values()), c


def main() -> None:
    import geonamescache

    table = load_modeling_table()
    train_ids = set(table.loc[table.split == "train", "project_id"])
    corpus = pd.read_parquet(OUT_PATH)
    sample = corpus[corpus.project_id.isin(train_ids)].sample(N_SAMPLE, random_state=SEED)
    raw = sample.sections.tolist()
    v1 = sample.sections_masked_v1.tolist()
    v2 = list_mask(v1, statement_country_names())

    countries, capitals = gazetteer()
    cap_set = {strip_accents(c) for c in capitals}
    lower_blob = strip_accents(" ".join(raw))
    cities = {
        strip_accents(c["name"])
        for c in geonamescache.GeonamesCache().get_cities().values()
        if c["population"] >= 100_000
    }
    cities = {
        c for c in cities if c not in cap_set and c.lower() not in lower_blob and c[0].isupper()
    }
    pats = {
        "pays + capitales": word_pattern([strip_accents(n) for n in countries] + list(cap_set)),
        "autres villes ≥ 100 000 hab. (indépendant)": word_pattern(cities),
    }
    n_chars = {"brut": sum(map(len, raw)), "v1": sum(map(len, v1)), "v2": sum(map(len, v2))}
    rows, examples = [], {}
    for label, pat in pats.items():
        row = {"compteur": label}
        for name, texts in [("brut", raw), ("v1", v1), ("v2", v2)]:
            n, c = count(pat, texts)
            row[name] = n
            row[f"{name}_per10k"] = 1e4 * n / n_chars[name]
            if name == "v2":
                examples[label] = c.most_common(10)
        rows.append(row)
    n_place = {k: sum(t.count("[PLACE]") for t in v) for k, v in [("v1", v1), ("v2", v2)]}

    lines = [
        "# Masquage des lieux : noms restants avant / après la liste fixe (généré)",
        "",
        f"Échantillon : {N_SAMPLE} projets tirés au hasard dans l'entraînement (graine {SEED}),",
        "sections clés (≈ 8 000 caractères). v1 = spaCy + pays des relevés IDA/IBRD ;",
        "v2 = v1 + liste fixe GeoNames (pays, capitales, accents ignorés) + anciens noms.",
        "",
        "| Compteur | brut | v1 | v2 | réduction v1 → v2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        red = 1 - r["v2"] / r["v1"] if r["v1"] else float("nan")
        lines.append(
            f"| {r['compteur']} | {r['brut']} | {r['v1']} | {r['v2']} | {red:.0%} |"
        )
    lines += [
        "",
        f"Jetons [PLACE] insérés : v1 = {n_place['v1']}, v2 = {n_place['v2']}.",
        "",
        "Noms encore présents en v2 (10 plus fréquents) :",
        "",
        *[f"- {k} : {v}" for k, v in examples.items()],
        "",
        "Lecture : le compteur « pays + capitales » est presque nul en v2 par construction ;",
        "le compteur indépendant (autres villes) dit ce que le masquage laisse passer.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
