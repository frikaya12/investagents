"""Génère le notebook Kaggle du bloc C et les métadonnées (jeu de données + notebook), privés.

- data/processed/kaggle_blocC/dataset-metadata.json : jeu de données privé
  `frikaya/investagents-blocc` (projects.jsonl + code du comité) ;
- kaggle/blocC/comite_blocC.ipynb + kernel-metadata.json : notebook privé, GPU + internet.

Le notebook : installe Ollama, lance UN serveur par GPU T4 (ports 11434 et 11435), télécharge
qwen2.5:7b, installe langgraph, extrait le code du comité, puis lance l'exécuteur
(committee_run/runner.py) qui écrit results.jsonl, recognition.jsonl et progress.json dans
/kaggle/working (sorties du notebook).

Lancer : `uv run python -m investagents.committee_run.make_kernel`, puis voir le README du bloc C
(kaggle/blocC/README.md) pour les commandes d'envoi.
"""

from __future__ import annotations

import json
from pathlib import Path

USER = "frikaya"
DATASET_SLUG = "investagents-blocc"
KERNEL_SLUG = "investagents-comite-blocc"
DATA_DIR = Path("data/processed/kaggle_blocC")
KERNEL_DIR = Path("kaggle/blocC")
MODEL = "qwen2.5:7b"
DEADLINE_HOURS = 11.0  # limite Kaggle 12 h ; ~15 min d'installation + marge

CELLS = [
    (
        "markdown",
        """# InvestAgents — bloc C : comité complet sur GPU (Qwen2.5 7B, Ollama, T4 x2)

Code : comité adapté de DIALECTIC (MIT). Documents : Banque mondiale ; notes IEG (CC BY 4.0) **non
présentes ici** : la vérité terrain reste sur le PC. Les agents ne voient que des phrases numérotées
des sections clés, identifiants et nom du projet masqués. Pas de recherche web pour les agents
(internet sert seulement à installer Ollama et télécharger le modèle).""",
    ),
    (
        "code",
        """import subprocess, time, os, glob, json
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
T0 = time.time()""",
    ),
    (
        "code",
        """# Installation d'Ollama (zstd nécessaire à l'archive de l'installeur)
!apt-get -qq update > /dev/null && apt-get -qq install -y zstd pciutils > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -3
!ollama --version""",
    ),
    (
        "code",
        f"""# Un serveur Ollama par GPU
import requests
gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.split()
gpus = [g for g in gpus if g == "GPU"]  # une ligne « GPU i: ... » par carte
N = max(1, len(gpus))
URLS, procs = [], []
for i in range(N):
    port = 11434 + i
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OLLAMA_HOST=f"127.0.0.1:{{port}}",
               OLLAMA_KEEP_ALIVE="24h", OLLAMA_NUM_PARALLEL="1")
    log = open(f"/kaggle/working/ollama_{{i}}.log", "w")
    procs.append(subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=log))
    URLS.append(f"http://127.0.0.1:{{port}}/api/chat")
for i in range(N):
    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{{11434 + i}}/api/version", timeout=2); break
        except Exception:
            time.sleep(2)
print("serveurs :", URLS)
!OLLAMA_HOST=127.0.0.1:11434 ollama pull {MODEL} 2>&1 | tail -2
for u in URLS:  # chargement du modèle sur chaque GPU
    t = time.time()
    msg = [{{"role": "user", "content": "Say OK"}}]
    body = {{"model": "{MODEL}", "messages": msg, "stream": False, "options": {{"num_predict": 5}}}}
    r = requests.post(u, json=body, timeout=900)
    print(u, r.status_code, r.json()["message"]["content"][:20], f"{{time.time()-t:.0f}} s")
!nvidia-smi --query-gpu=index,memory.used --format=csv""",
    ),
    (
        "code",
        """# Code du comité + dépendances
!pip -q install "langgraph>=1.0" 2>&1 | tail -1
import tarfile, sys
data_dir = os.path.dirname(glob.glob("/kaggle/input/**/projects.jsonl", recursive=True)[0])
print("données :", data_dir, os.listdir(data_dir))
# Kaggle décompresse lui-même les .tar.gz d'un jeu de données : on cherche le dossier du code
found = glob.glob("/kaggle/input/**/investagents/graph/committee.py", recursive=True)
if found:
    code_root = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
else:
    tarfile.open(os.path.join(data_dir, "investagents_code.tar.gz")).extractall("/tmp/code")
    code_root = "/tmp/code"
print("code :", code_root)
sys.path.insert(0, code_root)
import investagents.graph.committee  # vérifie que le code s'importe
print("code OK ; installation :", f"{(time.time()-T0)/60:.1f} min")""",
    ),
    (
        "code",
        f"""# Comité : essai 10 projets (temps par projet), reconnaissance 50 projets, puis tous.
# Sauvegarde après chaque projet ; arrêt propre avant la limite des 12 h.
deadline = {DEADLINE_HOURS} - (time.time() - T0) / 3600
urls = " ".join(URLS)
args = f"--data {{data_dir}} --out /kaggle/working --urls {{urls}} --deadline-hours {{deadline}}"
!cd /tmp && PYTHONPATH={{code_root}} python -u -m investagents.committee_run.runner {{args}} --model {MODEL} --rounds 1""",  # noqa: E501
    ),
    (
        "code",
        """print(open("/kaggle/working/progress.json").read())
for p in procs:
    p.terminate()""",
    ),
]


def notebook() -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"cell-{i}", "metadata": {},
                "source": src.splitlines(keepends=True)}  # fmt: skip
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": DATASET_SLUG,
                "id": f"{USER}/{DATASET_SLUG}",
                "licenses": [{"name": "CC-BY-4.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "comite_blocC.ipynb").write_text(
        json.dumps(notebook(), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{USER}/{KERNEL_SLUG}",
                "title": KERNEL_SLUG,
                "code_file": "comite_blocC.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{USER}/{DATASET_SLUG}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Écrit : {DATA_DIR}/dataset-metadata.json, {KERNEL_DIR}/")


if __name__ == "__main__":
    main()
