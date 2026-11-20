"""Extraction des sections clés d'un dossier d'évaluation, pour tenir dans 8 000 caractères.

Méthode simple et vérifiable (pas de LLM) : pour chaque thème (objectifs, description, risques,
analyse économique, mise en œuvre, leçons), on cherche les intitulés correspondants et on prend
la première occurrence située après les 8 premiers % du texte (pour sauter la table des
matières ; s'il n'y en a pas, la première tout court), puis un extrait de longueur égale pour
chaque thème trouvé. Les extraits sont des sous-chaînes exactes du texte
normalisé : une citation tirée d'un extrait se retrouve donc dans le document.

Masquage (règle anti-fuite) : identifiants de projet (P + 6 chiffres) et nom du projet remplacés.
"""

from __future__ import annotations

import re

THEMES = {
    "objectives": r"development objective|project objective",
    "description": r"project description|project components",
    "risks": r"(?:key |critical )?risks?(?: and mitigation)?",
    "economic": r"economic (?:and financial )?analysis|cost[- ]benefit",
    "implementation": r"implementation arrangements|institutional arrangements",
    "lessons": r"lessons (?:learned|learnt)",
}
TOC_FRACTION = 0.08


def normalize(text: str) -> str:
    """Espaces et sauts de ligne compactés : base commune pour extraits et vérification."""
    return re.sub(r"\s+", " ", text).strip()


def mask_identifiers(text: str, project_id: str | None = None, name: str | None = None) -> str:
    text = re.sub(r"\bP\d{6}\b", "[PROJECT_ID]", text)
    if name and len(name) > 5:
        text = re.sub(re.escape(name), "[PROJECT_NAME]", text, flags=re.IGNORECASE)
    return text


def extract_key_sections(text: str, max_chars: int = 8000) -> dict[str, str]:
    """Renvoie {thème: extrait} ; la somme des extraits ≤ max_chars."""
    t = normalize(text)
    starts = {}
    for theme, pattern in THEMES.items():
        hits = [m.start() for m in re.finditer(pattern, t, flags=re.IGNORECASE)]
        if not hits:
            continue
        after_toc = [h for h in hits if h > TOC_FRACTION * len(t)]
        starts[theme] = (after_toc or hits)[0]
    if not starts:  # aucun intitulé trouvé : début du document après la couverture
        return {"opening": t[:max_chars]}
    per = max_chars // len(starts)
    return {theme: t[s : s + per] for theme, s in starts.items()}


def sections_to_prompt(sections: dict[str, str]) -> str:
    return "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in sections.items())


# Fin de phrase : ponctuation forte suivie d'un espace puis d'une majuscule, d'un chiffre ou d'une
# parenthèse. Simple et suffisant pour des dossiers en anglais (OCR compris).
_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+(?=[A-Z0-9(])")
MIN_SENTENCE_CHARS = 25


def number_sentences(sections: dict[str, str]) -> list[dict]:
    """Découpe les sections en phrases numérotées S1, S2... (numérotation continue).

    Chaque phrase est une sous-chaîne exacte du texte donné aux agents. Les fragments trop
    courts (< 25 caractères : numéros de paragraphe, titres) sont rattachés à la phrase suivante.
    """
    out, n = [], 0
    for theme, text in sections.items():
        buffer = ""
        for piece in _SENTENCE_END.split(text):
            buffer = f"{buffer} {piece}".strip() if buffer else piece.strip()
            if len(buffer) >= MIN_SENTENCE_CHARS:
                n += 1
                out.append({"id": n, "theme": theme, "text": buffer})
                buffer = ""
        if buffer and out and out[-1]["theme"] == theme:
            out[-1]["text"] = f"{out[-1]['text']} {buffer}"
    return out


def sentences_to_prompt(sentences: list[dict]) -> str:
    lines, theme = [], None
    for s in sentences:
        if s["theme"] != theme:
            theme = s["theme"]
            lines.append(f"\n[{theme.upper()}]")
        lines.append(f"[S{s['id']}] {s['text']}")
    return "\n".join(lines).strip()
