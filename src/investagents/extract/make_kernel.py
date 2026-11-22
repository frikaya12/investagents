"""Génère le notebook Kaggle du bloc D (extracteur) et les métadonnées, privés.

Reprend les cellules d'installation du bloc C (Ollama, un serveur par GPU T4, qwen2.5:7b), puis
lance extract/runner.py. Deux modes :
- `--mode pilot` : 30 premiers projets (validation) ;
- `--mode full`  : tous les projets, dans l'ordre validation → test → entraînement, en reprenant
  les résultats d'une session précédente s'ils sont fournis comme source de notebook.

Lancer : `uv run python -m investagents.extract.make_kernel --mode pilot|full`
puis voir kaggle/blocD/README.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from investagents.committee_run.make_kernel import CELLS as CELLS_C
from investagents.committee_run.make_kernel import MODEL, USER

DATASET_SLUG = "investagents-blocd"
KERNEL_SLUG = "investagents-extracteur-blocd"
DATA_DIR = Path("data/processed/kaggle_blocD")
KERNEL_DIR = Path("kaggle/blocD")
DEADLINE_HOURS = 11.0

INTRO = (
    "markdown",
    """# InvestAgents — bloc D : agents extracteurs de facteurs de risque (Qwen2.5 7B, T4 x2)

Grille figée de 19 facteurs factuels ; citations par numéros de phrases ; « not_found » autorisé.
Vérité terrain (notes IEG) **absente** : elle reste sur le PC. Identifiants et nom des projets
masqués ; pas de recherche web pour les agents (internet sert à installer Ollama et le modèle).""",
)

CODE = (
    "code",
    """# Code de l'extracteur
import tarfile, sys
data_dir = os.path.dirname(glob.glob("/kaggle/input/**/projects.jsonl", recursive=True)[0])
print("données :", data_dir, os.listdir(data_dir))
found = glob.glob("/kaggle/input/**/investagents/extract/runner.py", recursive=True)
if found:
    code_root = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
else:
    tarfile.open(os.path.join(data_dir, "investagents_code.tar.gz")).extractall("/tmp/code")
    code_root = "/tmp/code"
sys.path.insert(0, code_root)
import investagents.extract.runner
print("code OK :", code_root, f"{(time.time()-T0)/60:.1f} min")""",
)


def run_cell(mode: str) -> tuple[str, str]:
    limit = "--limit 30" if mode == "pilot" else ""
    return (
        "code",
        f"""# Extraction ({mode}) ; sauvegarde après chaque projet ; arrêt propre avant 12 h
deadline = {DEADLINE_HOURS} - (time.time() - T0) / 3600
prev = " ".join(glob.glob("/kaggle/input/**/results.jsonl", recursive=True))
resume = f"--resume-from {{prev}}" if prev else ""
urls = " ".join(URLS)
args = f"--data {{data_dir}} --out /kaggle/working --urls {{urls}} --deadline-hours {{deadline}}"
args += f" {{resume}} --model {MODEL} {limit}"
!cd /tmp && PYTHONPATH={{code_root}} python -u -m investagents.extract.runner {{args}}
print(open("/kaggle/working/progress.json").read())
for p in procs:
    p.terminate()""",
    )


def notebook(mode: str) -> dict:
    cells_src = [INTRO, *CELLS_C[1:4], CODE, run_cell(mode)]
    cells = []
    for i, (kind, src) in enumerate(cells_src):
        cell = {"cell_type": kind, "id": f"cell-{i}", "metadata": {},
                "source": src.splitlines(keepends=True)}  # fmt: skip
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--resume-kernel", default=None,
                        help="notebook dont les sorties (results.jsonl) sont reprises")  # fmt: skip
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"title": DATASET_SLUG, "id": f"{USER}/{DATASET_SLUG}",
            "licenses": [{"name": "CC-BY-4.0"}]}  # fmt: skip
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "extracteur_blocD.ipynb").write_text(
        json.dumps(notebook(args.mode), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    kmeta = {
        "id": f"{USER}/{KERNEL_SLUG}", "title": KERNEL_SLUG,
        "code_file": "extracteur_blocD.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [f"{USER}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [args.resume_kernel] if args.resume_kernel else [],
    }  # fmt: skip
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print(f"Écrit : {KERNEL_DIR}/ (mode {args.mode})")


if __name__ == "__main__":
    main()
