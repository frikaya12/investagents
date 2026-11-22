"""Exécuteur de l'extracteur (bloc D) sur Kaggle : un serveur Ollama par GPU, un fil par serveur.

Ordre du fichier (validation, test, entraînement). Chaque résultat est ajouté à results.jsonl dès
qu'il est fini ; reprise : les clés déjà réussies (ici ou dans --resume-from) sont sautées. Arrêt
propre : on ne commence plus de projet si le temps écoulé + 3 fois le temps médian dépasse la
limite.

Lancer (sur Kaggle) :
`python -m investagents.extract.runner --data DIR --out /kaggle/working --urls URL1 URL2`
(option --limit 30 pour le pilote)
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

from investagents.agents.llm import call_json
from investagents.committee_run.runner import load_jsonl
from investagents.extract.extractor import extract


def run(projects, out: Path, urls, model, deadline_s, done=(), call=call_json) -> dict:
    t0 = time.time()
    lock = threading.Lock()
    durations: list[float] = []
    previous = load_jsonl(out / "results.jsonl")
    done = set(done) | {r["key"] for r in previous if r.get("factors")}
    queue: Queue = Queue()
    for p in sorted(projects, key=lambda p: p["order"]):
        if p["key"] not in done:
            queue.put(p)
    state = {"n_errors": 0, "stopped_for_time": False}

    def progress():
        prog = {"n_done": len(done), "n_total": len(projects), "n_errors": state["n_errors"],
                "median_seconds": statistics.median(durations) if durations else None,
                "elapsed_hours": round((time.time() - t0) / 3600, 2),
                "stopped_for_time": state["stopped_for_time"],
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        (out / "progress.json").write_text(json.dumps(prog, indent=2), encoding="utf-8")

    def worker(url):
        while True:
            typical = statistics.median(durations) if durations else 120.0
            if time.time() - t0 + 3 * typical > deadline_s:
                state["stopped_for_time"] = True
                return
            try:
                p = queue.get_nowait()
            except Empty:
                return
            rec = {"key": p["key"], "split": p["split"], "order": p["order"], "model": model}
            try:
                rec.update(extract(p, call, model=model, url=url))
            except Exception as exc:  # sortie invalide, serveur tombé : consigné, on continue
                rec.update(factors=None, error=repr(exc)[:500],
                           traceback=traceback.format_exc()[-1500:])  # fmt: skip
            rec["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with lock:
                with (out / "results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec.get("factors"):
                    done.add(p["key"])
                    durations.append(rec["seconds"])
                else:
                    state["n_errors"] += 1
                progress()
            print(f"[{rec['finished_at']}] {p['key']} ({p['split']}) {rec.get('seconds')} s "
                  f"{'ok' if rec.get('factors') else 'ÉCHEC ' + rec.get('error', '')}",
                  flush=True)  # fmt: skip

    with ThreadPoolExecutor(len(urls)) as pool:
        list(pool.map(worker, urls))
    progress()
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="/kaggle/working")
    parser.add_argument("--urls", nargs="+", default=["http://127.0.0.1:11434/api/chat"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--deadline-hours", type=float, default=11.0)
    parser.add_argument("--limit", type=int, default=None, help="pilote : N premiers projets")
    parser.add_argument("--resume-from", nargs="*", default=[])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    projects = sorted(load_jsonl(Path(args.data) / "projects.jsonl"), key=lambda p: p["order"])
    if args.limit:
        projects = projects[: args.limit]
    done = set()
    for path in args.resume_from:  # résultats d'une session précédente : recopiés et sautés
        with (out / "results.jsonl").open("a", encoding="utf-8") as f:
            for r in load_jsonl(Path(path)):
                if r.get("factors") and r["key"] not in done:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done.add(r["key"])
    state = run(projects, out, args.urls, args.model, args.deadline_hours * 3600, done)
    print(f"FIN : {state}", flush=True)


if __name__ == "__main__":
    main()
