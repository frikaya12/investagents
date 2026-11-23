"""Les chiffres du rapport final, de la vitrine et de la démo correspondent aux JSON de résultats.

1. Chaque {{clé}} du gabarit et du glossaire existe dans metrics.load().
2. Les fichiers générés (rapport_final.md, docs/rapport.html) sont à jour : les régénérer donne
   exactement le même contenu (personne ne les a modifiés à la main).
3. Contrôle indépendant : on relit les JSON ici, sans passer par metrics.py, et on vérifie que les
   chiffres clés apparaissent tels quels dans le rapport md et html.
4. La vitrine docs/index.html : ses chiffres (tableau DATA du graphique) égalent les JSON.
5. Les identifiants de commits cités comme preuves existent dans l'historique git.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from investagents.report import glossary
from investagents.report.build import TEMPLATE, render_html, render_md, template_keys
from investagents.report.metrics import ROOT, load

MD = ROOT / "reports" / "rapport_final.md"
HTML = ROOT / "docs" / "rapport.html"


def j(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def fr(x: float, nd: int = 3) -> str:
    """Mise en forme indépendante de metrics.num (pour ne pas tester le code avec lui-même)."""
    s = ("%." + str(nd) + "f") % abs(x)
    return ("\u2212" if x < 0 else "") + s.replace(".", ",")


def fr_ci(d: dict, signed: bool = False) -> str:
    v = fr(d["value"])
    if signed and d["value"] >= 0:
        v = "+" + v
    return f"{v} [{fr(d['ci_low'])} ; {fr(d['ci_high'])}]"


def test_every_template_key_exists():
    m = load()
    keys = template_keys(TEMPLATE.read_text(encoding="utf-8"))
    for t in glossary.TERMS:
        keys |= template_keys(t.simple + t.technique + t.exemple)
    missing = sorted(k for k in keys if k not in m.txt)
    assert not missing, f"clés absentes de metrics.load() : {missing}"


def test_glossary_has_required_terms():
    required = [
        "AUC-ROC", "AUC-PR", "taux de base", "Brier", "ECE", "calibration de Platt",
        "intervalle de confiance bootstrap", "écart apparié", "test de permutation", "valeur p",
        "validation croisée", "sur-apprentissage", "fuite de données", "découpage temporel",
        "coffre-fort", "décalage de domaine", "TF-IDF", "régression logistique",
        "gradient boosting", "embeddings", "LLM", "agent", "LangGraph", "citation vérifiée",
        "test de reconnaissance", "sac à dos (CP-SAT)", "IEG", "PAD", "critères OCDE-CAD",
        "frontière α",
    ]  # fmt: skip
    for term in required:
        t = glossary.find(term)
        assert t.simple and t.technique and t.exemple


def test_generated_report_files_are_up_to_date():
    m = load()
    md, svgs = render_md(m)
    assert MD.read_text(encoding="utf-8") == md, "relancer le build du rapport"
    assert HTML.read_text(encoding="utf-8") == render_html(m), "relancer le build du rapport"
    for name, svg in svgs.items():
        assert (ROOT / "reports" / "figures" / f"{name}.svg").read_text(encoding="utf-8") == svg


def _expected_numbers() -> list[str]:
    """Chiffres clés relus directement dans les JSON (sans metrics.py)."""
    bb = j("experiments/blocB/results.json")
    p2 = j("experiments/phase2/results.json")
    bc = j("experiments/blocC/models_results.json")
    m7 = j("experiments/blocC/m7_results.json")
    d6 = j("experiments/blocD/m6_results.json")
    v = j("experiments/vault/results.json")["all"]
    com = j("experiments/blocC/committee_summary.json")
    full = j("experiments/blocD/kaggle_full1/summary.json")
    pf = next(r for r in bb["portfolio"] if r["budget"] == 0.25)
    bt = {r["model"]: r for r in d6["backtest"]}
    return [
        fr_ci(bb["models"]["m1_text"]["auc_pr"]),
        fr_ci(bb["diffs"]["chosen - m1"]["auc_pr"], signed=True),
        fr_ci(p2["test_full"]["M1"]["auc_pr"]),
        fr_ci(bc["models"]["m3"]["ci"]["auc_pr"]),
        fr_ci(bc["diffs"]["m4 - texte"]["auc_pr"], signed=True),
        fr_ci(m7["diff_m7_texte"]["auc_pr"], signed=True),
        fr_ci(d6["diffs"]["m6 - texte"]["auc_pr"], signed=True),
        fr_ci(d6["diffs"]["m5 - m0"]["auc_pr"], signed=True),
        fr_ci(v["models"]["M1"]["ci"]["auc_pr"]),
        fr_ci(v["text_subset"]["M1-texte - M1"]["auc_pr"], signed=True),
        fr(pf["count_share_satisfactory"] * 100, 1) + "\u00a0%",
        fr(pf["random_count"] * 100, 1) + "\u00a0%",
        fr(bt["m6"]["count_share_satisfactory"] * 100, 1) + "\u00a0%",
        fr(bt["texte"]["count_share_satisfactory"] * 100, 1) + "\u00a0%",
        f"{com['quotes_found']:,} / {com['quotes_total']:,}".replace(",", "\u00a0"),
        fr(com["quotes_found"] / com["quotes_total"] * 100, 1) + "\u00a0%",
        f"{com['recognition']['recognized']} / {com['recognition']['n']}",
        fr(full["citation_valid_rate"] * 100, 1) + "\u00a0%",
    ]


@pytest.mark.parametrize("path", [MD, HTML], ids=["md", "html"])
def test_key_numbers_appear_in_report(path):
    content = path.read_text(encoding="utf-8")
    for s in _expected_numbers():
        assert s in content, f"« {s} » (lu dans les JSON) absent de {path.name}"


def test_metrics_raw_values_equal_json():
    m = load()
    bb = j("experiments/blocB/results.json")
    d6 = j("experiments/blocD/m6_results.json")
    assert m.raw["m1t_aucpr"] == bb["models"]["m1_text"]["auc_pr"]
    assert m.raw["m6_minus_text"] == d6["diffs"]["m6 - texte"]["auc_pr"]
    assert m.raw["m5_weights"] == d6["weights"]["m5"]


def test_showcase_index_numbers_match_json():
    """docs/index.html : chaque ligne du tableau DATA (AUC-PR, IC, taux de base) égale les JSON."""
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    block = re.search(r"const DATA = \[(.*?)\];", html, re.S).group(1)
    num = r"\s*([\d.]+)"
    rows = re.findall(rf"v:{num},\s*lo:{num},\s*hi:{num},\s*base:{num}", block)
    assert rows, "tableau DATA introuvable dans docs/index.html"
    m = load()
    from investagents.report.metrics import chart_auc

    expected = {(round(r["value"], 3), round(r["lo"], 3), round(r["hi"], 3), round(r["base"], 3))
                for r in chart_auc(m)}  # fmt: skip
    for row in rows:
        got = tuple(round(float(x), 3) for x in row)
        assert got in expected, f"ligne de la vitrine sans équivalent dans les JSON : {got}"


def test_commit_ids_cited_as_proofs_exist():
    if not (ROOT / ".git").exists():
        pytest.skip("pas de dépôt git")
    ids = set(re.findall(r"`([0-9a-f]{7})`", TEMPLATE.read_text(encoding="utf-8")))
    assert len(ids) >= 10
    for cid in ids:
        r = subprocess.run(["git", "cat-file", "-e", f"{cid}^{{commit}}"], cwd=ROOT,
                           capture_output=True)  # fmt: skip
        assert r.returncode == 0, f"commit {cid} introuvable"


def test_glossary_links_resolve():
    html = HTML.read_text(encoding="utf-8")
    targets = set(re.findall(r'class="gl" id="[^"]+" href="#([^"]+)"', html))
    assert targets, "aucun lien vers le glossaire"
    for t in targets:
        assert f'id="{t}"' in html
    for back in re.findall(r'class="back" href="#([^"]+)"', html):
        assert f'id="{back}"' in html


def test_pdf_exists():
    pdf = ROOT / "reports" / "rapport_final.pdf"
    assert pdf.exists() and pdf.stat().st_size > 50_000
    assert Path(pdf).read_bytes()[:4] == b"%PDF"


def test_factors_named_in_report_are_the_top_weights():
    """La phrase « Ce qu'il faut retenir » des poids cite les 5 plus gros poids positifs de M5."""
    from investagents.report.metrics import chart_factor_weights

    top5 = [r["factor"] for r in chart_factor_weights(load())[:5]]
    assert top5 == ["resettlement", "cofinanciers", "follow_on_phase", "env_A", "weak_capacity"]
    text = TEMPLATE.read_text(encoding="utf-8")
    for words in ("réinstallation", "cofinanceurs", "deuxième phase", 
                  "catégorie environnementale A",
                  "capacité faible"):  # fmt: skip
        assert words in text


def test_resume_prof_is_up_to_date_and_honest():
    """Résumé pour le prof : généré depuis les JSON, résultats négatifs présents, LLM exact."""
    from investagents.report.build import RESUME_OUT, RESUME_TEMPLATE, render_resume

    m = load()
    assert RESUME_OUT.read_text(encoding="utf-8") == render_resume(m), "relancer le build"
    keys = template_keys(RESUME_TEMPLATE.read_text(encoding="utf-8"))
    assert keys <= set(m.txt)
    text = RESUME_OUT.read_text(encoding="utf-8")
    v = j("experiments/vault/results.json")["all"]["models"]["M1"]["ci"]["auc_roc"]
    assert fr_ci(v) in text  # coffre-fort à 0,509, non atténué
    for word in ("Négatif", "Fragile", "Qwen2.5 7B via Ollama"):
        assert word in text


def test_no_paid_llm_mentioned():
    """Le rapport, le résumé, la vitrine et la démo ne mentionnent ni Gemini ni une API payante."""
    paths = [MD, HTML, ROOT / "docs" / "resume_prof.md", ROOT / "docs" / "index.html",
             *sorted((ROOT / "src" / "investagents" / "app").rglob("*.py"))]  # fmt: skip
    for p in paths:
        low = p.read_text(encoding="utf-8").lower()
        for bad in ("gemini", "openai", "gpt-4", "api payante utilisée"):
            assert bad not in low, f"{bad} dans {p.name}"
