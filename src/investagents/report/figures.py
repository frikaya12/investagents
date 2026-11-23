"""Graphiques du rapport final, en SVG écrit à la main (aucune dépendance de tracé).

Chaque fonction prend `Metrics` (les chiffres lus dans les JSON) et renvoie une chaîne SVG.
Le même SVG sert au Markdown (fichier dans reports/figures/), à la page HTML (en ligne) et au PDF.
Palette : celle de la vitrine (bleu principal, orange pour la 2e série, gris pour les références).
Chaque point porte une infobulle (<title>) avec la valeur exacte.
"""

from __future__ import annotations

from html import escape

from investagents.report.metrics import (
    Metrics,
    chart_auc,
    chart_backtest,
    chart_diffs,
    chart_factor_weights,
    chart_frontier,
    num,
    pct,
)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#7a7974"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#7a7974"
GRID, SURFACE = "#e7e6e1", "#fcfcfb"
FONT = "font-family=\"system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif\""


def _svg(w: int, h: int, body: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" role="img" aria-label="{escape(title)}" {FONT}>'
            f'<title>{escape(title)}</title><rect width="{w}" height="{h}" fill="{SURFACE}"/>'
            + "".join(body) + "</svg>")  # fmt: skip


def _text(x, y, s, size=12, fill=INK2, anchor="start", weight=400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" dominant-baseline="middle">'
            f"{escape(s)}</text>")  # fmt: skip


def _axis(sx, ticks, y_top, y_bot, fmt, label, w_mid) -> list[str]:
    out = []
    for t in ticks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y_top}" y2="{y_bot}" stroke="{GRID}"/>')
        out.append(_text(x, y_bot + 14, fmt(t), 11, MUTED, "middle"))
    out.append(_text(w_mid, y_bot + 34, label, 12, INK2, "middle"))
    return out


def _legend(items: list[tuple[str, str]], x: float, y: float) -> list[str]:
    out = []
    for shape, label in items:
        out.append(shape.format(x=x, y=y))
        out.append(_text(x + 14, y, label, 12, INK2))
        x += 26 + 7 * len(label)
    return out


DOT = '<circle cx="{x}" cy="{y}" r="5" fill="' + BLUE + '"/>'
HOLLOW = ('<circle cx="{x}" cy="{y}" r="5" fill="' + SURFACE + '" stroke="' + GREY
          + '" stroke-width="2"/>')
TICK = '<line x1="{x}" x2="{x}" y1="{y}" y2="{y}" stroke="' + GREY + '" stroke-width="3"/>'


def fig_auc(m: Metrics) -> str:
    rows = chart_auc(m)
    w, left, right, top, row_h = 760, 250, 24, 40, 34
    h = top + row_h * len(rows) + 50
    x0, x1 = 0.10, 0.45
    sx = lambda v: left + (v - x0) / (x1 - x0) * (w - left - right)  # noqa: E731
    body = _axis(sx, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], top - 8, top + row_h * len(rows),
                 lambda t: num(t, 2), "AUC-PR (classe « échec »), IC 95 %", (left + w - right) / 2)
    body += _legend([(DOT, "AUC-PR du modèle"),
                     ('<rect x="{x}" y="{y}" width="3" height="12" transform="translate(-1,-6)" '
                      f'fill="{GREY}"/>', "taux de base (hasard)")], left, 16)  # fmt: skip
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        body.append(_text(8, y - 6, r["model"], 12.5, INK, weight=600))
        body.append(_text(8, y + 8, f"{r['sample']} · n = {r['n']}", 11, MUTED))
        body.append(f'<line x1="{sx(r["lo"]):.1f}" x2="{sx(r["hi"]):.1f}" y1="{y}" y2="{y}" '
                    f'stroke="{BLUE}" stroke-width="2" stroke-linecap="round"/>')  # fmt: skip
        body.append(f'<rect x="{sx(r["base"]) - 1.5:.1f}" y="{y - 8}" width="3" height="16" '
                    f'fill="{GREY}"><title>taux de base {num(r["base"])}</title></rect>')
        body.append(f'<circle cx="{sx(r["value"]):.1f}" cy="{y}" r="5.5" fill="{BLUE}" '
                    f'stroke="{SURFACE}" stroke-width="2"><title>{escape(r["model"])} : '
                    f'{num(r["value"])} [{num(r["lo"])} ; {num(r["hi"])}], taux de base '
                    f'{num(r["base"])}</title></circle>')  # fmt: skip
    return _svg(w, h, body, "AUC-PR des modèles au test, avec IC 95 % et taux de base")


def fig_diffs(m: Metrics) -> str:
    rows = chart_diffs(m)
    w, left, right, top, row_h = 760, 330, 24, 40, 34
    h = top + row_h * len(rows) + 50
    x0, x1 = -0.10, 0.15
    sx = lambda v: left + (v - x0) / (x1 - x0) * (w - left - right)  # noqa: E731
    body = _axis(sx, [-0.1, -0.05, 0, 0.05, 0.1, 0.15], top - 8, top + row_h * len(rows),
                 lambda t: num(t, 2, signed=True), "Écart apparié d'AUC-PR, IC 95 %",
                 (left + w - right) / 2)  # fmt: skip
    body.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top - 8}" '
                f'y2="{top + row_h * len(rows)}" stroke="{INK2}" stroke-width="1.5"/>')
    body += _legend([(DOT, "IC entièrement au-dessus de 0 : apport démontré"),
                     (HOLLOW, "IC contient 0 : non démontré")], 12, 16)  # fmt: skip
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        sure = r["lo"] > 0
        body.append(_text(8, y - 6, r["label"], 12.5, INK, weight=600))
        body.append(_text(8, y + 8, r["question"], 11, MUTED))
        color = BLUE if sure else GREY
        body.append(f'<line x1="{sx(max(r["lo"], x0)):.1f}" x2="{sx(min(r["hi"], x1)):.1f}" '
                    f'y1="{y}" y2="{y}" stroke="{color}" stroke-width="2" '
                    'stroke-linecap="round"/>')  # fmt: skip
        fill = BLUE if sure else SURFACE
        body.append(f'<circle cx="{sx(r["value"]):.1f}" cy="{y}" r="5.5" fill="{fill}" '
                    f'stroke="{color}" stroke-width="2"><title>{escape(r["label"])} : '
                    f'{num(r["value"], 3, True)} [{num(r["lo"])} ; {num(r["hi"])}]</title>'
                    "</circle>")  # fmt: skip
    return _svg(w, h, body, "Écarts appariés d'AUC-PR (forest plot)")


def fig_backtest(m: Metrics) -> str:
    d = chart_backtest(m)
    b, dd = d["blocB"], d["blocD"]
    rows = [
        ("Bloc B · M1-texte", f"pool de {b['n_pool']} projets", b["ours"], b["ci"],
         (b["random_lo"], b["random_mean"], b["random_hi"])),
        ("Bloc B · contrôle M0", "taille des projets seule", b["m0"], None,
         (b["random_lo"], b["random_mean"], b["random_hi"])),
        ("Bloc D · M6", f"pool de {dd['n_pool']} projets", dd["m6"], dd["m6_ci"],
         (None, dd["random_mean"], None)),
        ("Bloc D · M1-texte", f"pool de {dd['n_pool']} projets", dd["text"], dd["text_ci"],
         (None, dd["random_mean"], None)),
    ]  # fmt: skip
    w, left, right, top, row_h = 760, 250, 24, 44, 40
    h = top + row_h * len(rows) + 50
    x0, x1 = 0.65, 0.95
    sx = lambda v: left + (v - x0) / (x1 - x0) * (w - left - right)  # noqa: E731
    body = _axis(sx, [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], top - 8, top + row_h * len(rows),
                 lambda t: pct(t, 0), "Part de projets réussis du portefeuille (25 % du budget)",
                 (left + w - right) / 2)  # fmt: skip
    body += _legend([(DOT, "portefeuille (IC 95 %)"),
                     ('<rect x="{x}" y="{y}" width="12" height="10" transform="translate(-6,-5)" '
                      'fill="#e2e1dc"/>', "95 % des portefeuilles tirés au hasard"),
                     (TICK.replace('y1="{y}"', 'y1="{y}" transform="translate(0,-6)"')
                      .replace('y2="{y}"', 'y2="{y}"'), "")], left - 40, 16)  # fmt: skip
    for i, (name, sub, v, c, (lo, mean, hi)) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        body.append(_text(8, y - 6, name, 12.5, INK, weight=600))
        body.append(_text(8, y + 8, sub, 11, MUTED))
        if lo is not None:
            body.append(f'<rect x="{sx(lo):.1f}" y="{y - 9}" width="{sx(hi) - sx(lo):.1f}" '
                        f'height="18" rx="3" fill="#e2e1dc"><title>hasard : 95 % des tirages '
                        f'entre {pct(lo)} et {pct(hi)}</title></rect>')  # fmt: skip
        body.append(f'<rect x="{sx(mean) - 1.5:.1f}" y="{y - 11}" width="3" height="22" '
                    f'fill="{GREY}"><title>hasard, moyenne {pct(mean)}</title></rect>')
        if c is not None:
            body.append(f'<line x1="{sx(c[0]):.1f}" x2="{sx(c[1]):.1f}" y1="{y}" y2="{y}" '
                        f'stroke="{BLUE}" stroke-width="2" stroke-linecap="round"/>')
            body.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="5.5" fill="{BLUE}" '
                        f'stroke="{SURFACE}" stroke-width="2"><title>{escape(name)} : {pct(v)} '
                        f"[{pct(c[0])} ; {pct(c[1])}]</title></circle>")  # fmt: skip
        else:
            body.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="5.5" fill="{SURFACE}" '
                        f'stroke="{GREY}" stroke-width="2"><title>{escape(name)} : {pct(v)}'
                        "</title></circle>")  # fmt: skip
    return _svg(w, h, body, "Backtest du portefeuille contre les tirages au hasard")


def fig_frontier(m: Metrics) -> str:
    d = chart_frontier(m)
    w, h, left, right, top, bottom = 760, 380, 70, 130, 30, 60
    x0, x1, y0, y1 = 6.2, 7.6, 0.70, 0.95
    sx = lambda v: left + (v - x0) / (x1 - x0) * (w - left - right)  # noqa: E731
    sy = lambda v: top + (y1 - v) / (y1 - y0) * (h - top - bottom)  # noqa: E731
    body = []
    for t in (6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6):
        body.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" y2="{h - bottom}" '
                    f'stroke="{GRID}"/>')
        body.append(_text(sx(t), h - bottom + 14, num(t, 1), 11, MUTED, "middle"))
    for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        body.append(f'<line x1="{left}" x2="{w - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" '
                    f'stroke="{GRID}"/>')
        body.append(_text(left - 8, sy(t), pct(t, 0), 11, MUTED, "end"))
    body.append(_text((left + w - right) / 2, h - 18,
                      "Note moyenne « impact et durabilité » du portefeuille (Juge, sur 10)", 12,
                      INK2, "middle"))  # fmt: skip
    body.append(_text(left, 12, "Part de projets réussis (issue IEG)", 12, INK2))
    for name, color in (("validation", BLUE), ("test", ORANGE)):
        pts = d[name]["points"]
        path = " ".join(f"{sx(p['impact']):.1f},{sy(p['success']):.1f}" for p in pts)
        body.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for p in pts:
            body.append(f'<circle cx="{sx(p["impact"]):.1f}" cy="{sy(p["success"]):.1f}" r="5" '
                        f'fill="{color}" stroke="{SURFACE}" stroke-width="2"><title>{name}, α = '
                        f'{num(p["alpha"], 2)} : {pct(p["success"])} de réussite, impact '
                        f'{num(p["impact"], 2)}</title></circle>')  # fmt: skip
            dy = -10 if name == "validation" else 13  # étiquettes au-dessus / au-dessous
            dx, anchor = (7, "start") if name == "validation" else (-7, "end")
            body.append(_text(sx(p["impact"]) + dx, sy(p["success"]) + dy,
                              f"α={num(p['alpha'], 2).rstrip('0').rstrip(',')}", 10, MUTED,
                              anchor))
        r = d[name]["random"]
        x, y = sx(r["impact"]), sy(r["success"])
        body.append(f'<path d="M{x - 5},{y - 5}L{x + 5},{y + 5}M{x - 5},{y + 5}L{x + 5},{y - 5}" '
                    f'stroke="{color}" stroke-width="2"><title>hasard ({name}) : '
                    f'{pct(r["success"])}, impact {num(r["impact"], 2)}</title></path>')
        last = pts[-1]
        body.append(_text(w - right + 10, sy(pts[0]["success"]), f"● {name}", 12, color,
                          weight=600))
        del last
    body.append(_text(w - right + 10, h - bottom - 10, "× hasard", 12, INK2))
    return _svg(w, h, body, "Frontière α : réussite contre note d'impact")


def fig_weights(m: Metrics) -> str:
    rows = chart_factor_weights(m)
    w, left, right, top, row_h = 760, 300, 40, 40, 22
    h = top + row_h * len(rows) + 50
    lim = max(abs(r["weight"]) for r in rows) * 1.15
    sx = lambda v: left + (v + lim) / (2 * lim) * (w - left - right)  # noqa: E731
    body = _axis(sx, [-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06], top - 8, top + row_h * len(rows),
                 lambda t: num(t, 2, signed=True),
                 "Poids dans M5 (variables standardisées, forte régularisation)",
                 (left + w - right) / 2)  # fmt: skip
    body += _legend([('<rect x="{x}" y="{y}" width="12" height="10" transform="translate(-6,-5)" '
                      f'fill="{ORANGE}"/>', "associé à plus d'échecs"),
                     ('<rect x="{x}" y="{y}" width="12" height="10" transform="translate(-6,-5)" '
                      f'fill="{BLUE}"/>', "associé à moins d'échecs")], left, 16)  # fmt: skip
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        v = r["weight"]
        color = ORANGE if v > 0 else BLUE
        x_a, x_b = sorted((sx(0), sx(v)))
        body.append(_text(left - 8, y, r["label"], 11.5, INK, "end"))
        body.append(f'<rect x="{x_a:.1f}" y="{y - 7}" width="{max(x_b - x_a, 1):.1f}" height="14" '
                    f'rx="2" fill="{color}"><title>{escape(r["label"])} : '
                    f'{num(v, 4, True)}</title></rect>')  # fmt: skip
    body.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top - 8}" '
                f'y2="{top + row_h * len(rows)}" stroke="{INK2}"/>')
    return _svg(w, h, body, "Poids des facteurs extraits dans M5")


def _box(x, y, w, h, title, sub, fill="#e8f0fb", stroke=BLUE) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"/>'
            + _text(x + w / 2, y + h / 2 - (8 if sub else 0), title, 12.5, INK, "middle", 600)
            + (_text(x + w / 2, y + h / 2 + 10, sub, 10.5, INK2, "middle") if sub else ""))


def _arrow(x1, y1, x2, y2) -> str:
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.5" '
            'marker-end="url(#arr)"/>')


DEFS = (f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="{INK2}"/></marker></defs>')


def fig_pipeline(m: Metrics) -> str:
    steps = [
        ("Données", "IEG + dossiers ex ante"),
        ("Anti-fuite", "dates, masquage"),
        ("ML", "M1, M1-texte"),
        ("Agents", "juges + extracteurs"),
        ("Sélection", "sac à dos, budget"),
    ]
    w, h, bw, gap = 760, 120, 124, 28
    body = [DEFS]
    for i, (t, s) in enumerate(steps):
        x = 10 + i * (bw + gap)
        body.append(_box(x, 30, bw, 56, t, s))
        if i:
            body.append(_arrow(x - gap + 2, 58, x - 3, 58))
    body.append(_text(w / 2, 108, f"Vérité terrain : note IEG (échec = MU/U/HU) · "
                      f"{m.txt['n_exploitable']} projets avec dossier exploitable", 11.5, MUTED,
                      "middle"))  # fmt: skip
    return _svg(w, h, body, "Pipeline InvestAgents")


def fig_committee(m: Metrics) -> str:
    del m
    w, h = 760, 250
    body = [DEFS]
    body.append(_box(10, 20, 150, 56, "Dossier ex ante", "masqué, daté avant"))
    body.append(_box(200, 20, 150, 56, "Phrases clés", "numérotées [S1]…[Sn]"))
    body.append(_box(390, 10, 170, 36, "Avocat Pour", "", "#e8f0fb"))
    body.append(_box(390, 54, 170, 36, "Avocat Contre", "", "#fdeee7", ORANGE))
    body.append(_box(600, 20, 150, 56, "Critique", "fort / faible / non étayé"))
    body.append(_box(600, 130, 150, 56, "Raffinement", "1 tour de réécriture"))
    body.append(_box(390, 130, 170, 56, "Juge", "5 critères, grille à ancres"))
    body.append(_box(200, 130, 150, 56, "Vérification", "citations en Python", "#eef7f2",
                     "#1baf7a"))
    body.append(_box(10, 130, 150, 56, "Notes + arguments", "sourcés, JSON validé"))
    body += [_arrow(160, 48, 197, 48), _arrow(350, 40, 387, 28), _arrow(350, 56, 387, 72),
             _arrow(560, 28, 597, 40), _arrow(560, 72, 597, 58), _arrow(675, 76, 675, 127),
             _arrow(600, 158, 563, 158), _arrow(390, 158, 353, 158), _arrow(200, 158, 163, 158)]
    body.append(_text(w / 2, 222, "Pour chacun des 5 critères : pertinence, efficience "
                      "économique, faisabilité, risque, impact et durabilité", 11.5, MUTED,
                      "middle"))  # fmt: skip
    return _svg(w, h, body, "Architecture du comité d'agents juges")


FIGURES = {
    "pipeline": fig_pipeline,
    "comite": fig_committee,
    "auc": fig_auc,
    "ecarts": fig_diffs,
    "backtest": fig_backtest,
    "frontiere": fig_frontier,
    "poids": fig_weights,
}
