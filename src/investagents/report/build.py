"""Construit le rapport final en trois versions à partir d'un seul gabarit et des JSON de résultats.

- reports/rapport_final.md  (+ reports/figures/*.svg)
- docs/rapport.html         (page web : sommaire latéral, glossaire cliquable, graphiques en ligne)
- reports/rapport_final.pdf (impression de la page HTML par Microsoft Edge sans interface)

Syntaxe du gabarit (src/investagents/report/rapport_template.md) :
- `{{clé}}`            -> chiffre lu par metrics.load() (erreur si la clé n'existe pas) ;
- `[[terme]]` ou `[[terme|texte affiché]]` -> lien vers le glossaire (erreur si terme absent) ;
- `@@fig:nom@@`        -> graphique de figures.FIGURES ;
- `@@glossary@@`       -> le glossaire complet.

Lancer : `uv run python -m investagents.report.build` (option --no-pdf pour sauter le PDF).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt

from investagents.report import glossary
from investagents.report.figures import FIGURES
from investagents.report.metrics import ROOT, Metrics, load

TEMPLATE = Path(__file__).with_name("rapport_template.md")
MD_OUT = ROOT / "reports" / "rapport_final.md"
FIG_DIR = ROOT / "reports" / "figures"
HTML_OUT = ROOT / "docs" / "rapport.html"
PDF_OUT = ROOT / "reports" / "rapport_final.pdf"
EDGE = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

KEY_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
FIG_RE = re.compile(r"^@@fig:([a-z]+)@@$", re.M)
H2_RE = re.compile(r"^## (\d+)\. (.+)$", re.M)


def fill(text: str, m: Metrics) -> str:
    """Remplace chaque {{clé}} par le chiffre mis en forme ; erreur si une clé est inconnue."""

    def rep(match: re.Match) -> str:
        key = match.group(1)
        if key not in m.txt:
            raise KeyError(f"clé inconnue dans le gabarit : {key}")
        return m.txt[key]

    return KEY_RE.sub(rep, text)


def template_keys(text: str) -> set[str]:
    return set(KEY_RE.findall(text))


def headings(text: str) -> list[tuple[str, str]]:
    return [(f"s-{n}", f"{n}. {t}") for n, t in H2_RE.findall(text)]


def _links(text: str, html: bool) -> str:
    """[[terme|affiché]] -> lien vers le glossaire (ancre de retour sur la 1re occurrence)."""
    seen: dict[str, int] = {}

    def rep(match: re.Match) -> str:
        term = glossary.find(match.group(1).strip())
        shown = (match.group(2) or match.group(1)).strip()
        k = seen.get(term.anchor, 0)
        seen[term.anchor] = k + 1
        ref = f"ref-{term.anchor}" + (f"-{k}" if k else "")
        if html:
            return f'<a class="gl" id="{ref}" href="#{term.anchor}">{escape(shown)}</a>'
        anchor = f'<a id="{ref}"></a>' if k == 0 else ""
        return f"{anchor}[{shown}](#{term.anchor})"

    return LINK_RE.sub(rep, text)


def _glossary_md(m: Metrics, linked: set[str]) -> str:
    out = []
    for t in sorted(glossary.TERMS, key=lambda t: t.term.lower()):
        back = f"#ref-{t.anchor}" if t.anchor in linked else "#sommaire"
        out += [
            f'<a id="{t.anchor}"></a>',
            f"### {t.term}",
            "",
            f"- **En simple :** {fill(t.simple, m)}",
            f"- **En technique :** {fill(t.technique, m)}",
            f"- **Dans notre projet :** {fill(t.exemple, m)}",
            "",
            f"[↩ retour au texte]({back})",
            "",
        ]
    return "\n".join(out)


def _glossary_html(m: Metrics, linked: set[str]) -> str:
    out = ['<div class="glossary">']
    for t in sorted(glossary.TERMS, key=lambda t: t.term.lower()):
        back = f"#ref-{t.anchor}" if t.anchor in linked else "#sommaire"
        out.append(
            f'<section class="term" id="{t.anchor}"><h3>{escape(t.term)}</h3><dl>'
            f"<dt>En simple</dt><dd>{escape(fill(t.simple, m))}</dd>"
            f"<dt>En technique</dt><dd>{escape(fill(t.technique, m))}</dd>"
            f"<dt>Dans notre projet</dt><dd>{escape(fill(t.exemple, m))}</dd></dl>"
            f'<a class="back" href="{back}" data-term="{t.anchor}">↩ retour au texte</a>'
            "</section>"
        )
    out.append("</div>")
    return "\n".join(out)


def _sommaire_md(heads: list[tuple[str, str]]) -> str:
    lines = ['<a id="sommaire"></a>', "**Sommaire**", ""]
    lines += [f"- [{t}](#{a})" for a, t in heads]
    return "\n".join(lines) + "\n"


def render_md(m: Metrics) -> tuple[str, dict[str, str]]:
    """Renvoie (markdown, {nom: svg})."""
    src = TEMPLATE.read_text(encoding="utf-8")
    heads = headings(src)
    text = fill(src, m)
    linked = {glossary.find(a.strip()).anchor for a, _ in LINK_RE.findall(text)}
    text = _links(text, html=False)
    text = H2_RE.sub(lambda g: f'<a id="s-{g.group(1)}"></a>\n\n## {g.group(1)}. {g.group(2)}',
                     text)  # fmt: skip
    first_h2 = text.index('<a id="s-0">')
    text = text[:first_h2] + _sommaire_md(heads) + "\n" + text[first_h2:]
    svgs = {name: FIGURES[name](m) for name in FIG_RE.findall(text)}
    text = FIG_RE.sub(lambda g: f"![{g.group(1)}](figures/{g.group(1)}.svg)", text)
    text = text.replace("@@glossary@@", _glossary_md(m, linked))
    header = ("<!-- Fichier généré par `uv run python -m investagents.report.build` : ne pas "
              "modifier à la main (modifier src/investagents/report/rapport_template.md). -->\n")
    return header + text, svgs


STATIC = Path(__file__).with_name("static")
CSS = (STATIC / "rapport.css").read_text(encoding="utf-8")

JS = (STATIC / "rapport.js").read_text(encoding="utf-8")


def render_html(m: Metrics) -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    heads = headings(src)
    text = fill(src, m)
    linked = {glossary.find(a.strip()).anchor for a, _ in LINK_RE.findall(text)}
    text = _links(text, html=True)
    text = H2_RE.sub(lambda g: f'<h2 id="s-{g.group(1)}">{g.group(1)}. {escape(g.group(2))}</h2>',
                     text)  # fmt: skip
    text = FIG_RE.sub(lambda g: f"<figure>{FIGURES[g.group(1)](m)}</figure>", text)
    text = text.replace("@@glossary@@", "GLOSSARY_PLACEHOLDER")
    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")
    body = md.render(text).replace("<p>GLOSSARY_PLACEHOLDER</p>", _glossary_html(m, linked))
    first_h2 = body.index('<h2 id="s-0">')
    somm = ('<div class="sommaire" id="sommaire"><strong>Sommaire</strong><ol start="0">'
            + "".join(f'<li><a href="#{a}">{escape(t.split(". ", 1)[1])}</a></li>'
                      for a, t in heads) + "</ol></div>")  # fmt: skip
    body = body[:first_h2] + somm + body[first_h2:]
    toc = "".join(f'<a href="#{a}">{escape(t)}</a>' for a, t in heads)
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InvestAgents : rapport final</title>
<meta name="description" content="Rapport final du projet InvestAgents : agents d'IA et machine
learning pour évaluer et sélectionner des projets de la Banque mondiale.">
<!-- Fichier généré par `uv run python -m investagents.report.build` (ne pas modifier). -->
<style>{CSS}</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Sommaire">
<div class="brand">InvestAgents</div>
<div class="sub">Rapport final · sujet n°10</div>
{toc}
<div class="links"><a href="index.html">← Vitrine du projet</a>
<a href="https://github.com/frikaya12/investagents">Code sur GitHub</a></div>
</nav>
<main>
{body}
</main>
</div>
<script>{JS}</script>
</body>
</html>
"""


SHOWCASE = ROOT / "docs" / "index.html"
DATA_RE = re.compile(r"const DATA = \[.*?\];", re.S)


def showcase_data(m: Metrics) -> str:
    """Tableau DATA du graphique de la vitrine, écrit depuis les JSON (comme le rapport)."""
    from investagents.report.metrics import chart_auc

    rows = [
        f'  {{ m: "{r["short"]}", d: "{r["desc"]}", n: {r["n"]}, v: {r["value"]:.3f}, '
        f'lo: {r["lo"]:.3f}, hi: {r["hi"]:.3f}, base: {r["base"]:.3f} }},'
        for r in chart_auc(m)
    ]
    return "const DATA = [\n" + "\n".join(rows) + "\n];"


def update_showcase(m: Metrics) -> str:
    html = SHOWCASE.read_text(encoding="utf-8")
    if not DATA_RE.search(html):
        raise ValueError("bloc « const DATA = [...] » introuvable dans docs/index.html")
    return DATA_RE.sub(lambda _: showcase_data(m), html, count=1)


def find_edge() -> Path | None:
    for p in EDGE:
        if p.exists():
            return p
    found = shutil.which("msedge")
    return Path(found) if found else None


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    edge = find_edge()
    if edge is None:
        raise RuntimeError("Microsoft Edge introuvable : PDF non généré")
    subprocess.run(
        [str(edge), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
        check=True, capture_output=True, timeout=180,
    )  # fmt: skip
    if not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
        raise RuntimeError("PDF vide ou absent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    m = load()
    md, svgs = render_md(m)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, svg in svgs.items():
        (FIG_DIR / f"{name}.svg").write_text(svg, encoding="utf-8")
    MD_OUT.write_text(md, encoding="utf-8")
    HTML_OUT.write_text(render_html(m), encoding="utf-8")
    SHOWCASE.write_text(update_showcase(m), encoding="utf-8")
    print(f"écrit : {MD_OUT.relative_to(ROOT)}, {HTML_OUT.relative_to(ROOT)}, {len(svgs)} figures")
    if not args.no_pdf:
        render_pdf(HTML_OUT, PDF_OUT)
        print(f"écrit : {PDF_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
