"""Exécuteur du comité complet sur Kaggle (GPU T4 x2, un serveur Ollama par GPU).

Étapes, dans l'ordre :
1. essai : les 10 premiers projets (ordre alterné validation / test), temps par projet affiché ;
2. test de reconnaissance sur les 50 premiers projets (1 appel court par projet) ;
3. tous les autres projets.
Chaque résultat est ajouté à `results.jsonl` dès qu'il est fini (aucune perte si la session
s'arrête). Reprise : les clés déjà présentes dans `results.jsonl` (ou dans les dossiers donnés
par --resume-from) sont sautées. Arrêt propre : on ne commence plus de projet si le temps écoulé
plus 2 fois le temps médian par projet dépasse --deadline-hours.

Le modèle ne reçoit que les phrases numérotées (identifiants et nom masqués) ; aucune vérité
terrain n'est présente sur Kaggle.

Lancer (sur Kaggle) :
`python -m investagents.committee_run.runner --data DIR --out /kaggle/working --urls URL1 URL2`
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
from queue import Queue

from investagents.agents.llm import LLMOutputError, call_json
from investagents.agents.sections import sentences_to_prompt
from investagents.debate.prompts import RECOGNITION_SYSTEM, RECOGNITION_USER
from investagents.schemas.committee import RecognitionOutput

N_TRIAL = 10
N_RECOGNITION = 50
NUM_CTX = 12288  # document + arguments + critique tiennent sans troncature


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # dernière ligne coupée par un arrêt brutal
                continue
    return out


class Runner:
    def __init__(self, projects, out_dir: Path, urls, model, rounds, deadline_s, graph=None,
                 done_keys=()):  # fmt: skip
        self.projects = sorted(projects, key=lambda p: p["order"])
        self.out = out_dir
        self.urls = list(urls)
        self.model = model
        self.rounds = rounds
        self.t0 = time.time()
        self.deadline_s = deadline_s
        self.lock = threading.Lock()
        self.durations: list[float] = []
        previous = load_jsonl(out_dir / "results.jsonl")
        # un projet en échec lors d'une session précédente est retenté
        self.done = set(done_keys) | {r["key"] for r in previous if r.get("judge") is not None}
        self.n_errors = 0
        self.stopped_for_time = False
        if graph is None:
            from investagents.graph.committee import build_full_committee

            graph = build_full_committee()
        self.graph = graph

    # ------------------------------------------------------------ temps
    def time_left_ok(self) -> bool:
        elapsed = time.time() - self.t0
        typical = statistics.median(self.durations) if self.durations else 600.0
        return elapsed + 2 * typical < self.deadline_s

    # ------------------------------------------------------------ un projet
    def run_one(self, project: dict, url: str) -> dict:
        t = time.time()
        rec = {"key": project["key"], "split": project["split"], "order": project["order"],
               "model": self.model, "rounds": self.rounds, "url": url}  # fmt: skip
        try:
            state = self.graph.invoke(
                {
                    "masked_id": project["key"],
                    "sentences": project["sentences"],
                    "model": self.model,
                    "ollama_url": url,
                    "rounds": self.rounds,
                    "num_ctx": NUM_CTX,
                }
            )
            cites = state.get("citations", [])
            rec.update(
                initial_debate=state.get("initial_debate"),
                rounds_history=state.get("rounds_history", []),
                debate=state.get("debate"),
                judge=state.get("judge"),
                citations=cites,
                citation_valid_rate=(sum(c["valid"] for c in cites) / len(cites))
                if cites
                else None,
                llm_stats=state.get("llm_stats", {}),
                errors=state.get("errors", []),
            )
        except Exception as exc:  # serveur tombé, délai… : consigné, on continue
            rec.update(judge=None, errors=[f"exception : {exc!r}"],
                       traceback=traceback.format_exc()[-2000:])  # fmt: skip
        rec["seconds"] = round(time.time() - t, 1)
        rec["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            with (self.out / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.done.add(project["key"])
            if rec.get("judge") is not None:
                self.durations.append(rec["seconds"])
            if rec["errors"]:
                self.n_errors += 1
            self.write_progress()
        print(
            f"[{rec['finished_at']}] {rec['key']} ({rec['split']}) {rec['seconds']} s "
            f"juge={'ok' if rec.get('judge') else 'ÉCHEC'} "
            f"citations={rec.get('citation_valid_rate')} erreurs={len(rec['errors'])}",
            flush=True,
        )
        return rec

    def write_progress(self) -> None:
        d = self.durations
        prog = {
            "n_done": len(self.done),
            "n_total": len(self.projects),
            "n_with_errors": self.n_errors,
            "median_seconds_per_project": statistics.median(d) if d else None,
            "mean_seconds_per_project": statistics.mean(d) if d else None,
            "elapsed_hours": round((time.time() - self.t0) / 3600, 2),
            "n_servers": len(self.urls),
            "stopped_for_time": self.stopped_for_time,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.out / "progress.json").write_text(json.dumps(prog, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ un lot en parallèle
    def run_batch(self, projects: list[dict]) -> None:
        todo = [p for p in projects if p["key"] not in self.done]
        queue: Queue = Queue()
        for p in todo:
            queue.put(p)

        def worker(url):
            while not queue.empty():
                if not self.time_left_ok():
                    self.stopped_for_time = True
                    return
                try:
                    p = queue.get_nowait()
                except Exception:
                    return
                self.run_one(p, url)

        with ThreadPoolExecutor(len(self.urls)) as pool:
            list(pool.map(worker, self.urls))

    # ------------------------------------------------------------ reconnaissance
    def recognition(self, n: int = N_RECOGNITION) -> None:
        path = self.out / "recognition.jsonl"
        done = {r["key"] for r in load_jsonl(path)}
        todo = [p for p in self.projects[:n] if p["key"] not in done]
        queue: Queue = Queue()
        for p in todo:
            queue.put(p)

        def worker(url):
            while not queue.empty():
                try:
                    p = queue.get_nowait()
                except Exception:
                    return
                t = time.time()
                rec = {"key": p["key"], "split": p["split"]}
                try:
                    out, stats = call_json(
                        RECOGNITION_SYSTEM,
                        RECOGNITION_USER.format(document=sentences_to_prompt(p["sentences"])),
                        RecognitionOutput,
                        model=self.model,
                        url=url,
                        temperature=0.0,
                        num_predict=300,
                        num_ctx=NUM_CTX,
                    )
                    rec.update(out.model_dump(), llm_stats=stats)
                except (LLMOutputError, Exception) as exc:
                    rec["error"] = repr(exc)
                rec["seconds"] = round(time.time() - t, 1)
                with self.lock, path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(len(self.urls)) as pool:
            list(pool.map(worker, self.urls))
        print(f"Reconnaissance : {len(load_jsonl(path))} projets", flush=True)

    # ------------------------------------------------------------ tout
    def run_all(self) -> None:
        print(f"Projets : {len(self.projects)}, déjà faits : {len(self.done)}", flush=True)
        self.run_batch(self.projects[:N_TRIAL])
        d = self.durations
        if d:
            print(
                f"ESSAI {N_TRIAL} projets : médiane {statistics.median(d):.0f} s/projet, "
                f"min {min(d):.0f}, max {max(d):.0f} ; {len(self.urls)} serveurs en parallèle "
                f"→ débit ≈ {3600 * len(self.urls) / statistics.median(d):.0f} projets/h",
                flush=True,
            )
        self.recognition()
        self.run_batch(self.projects[N_TRIAL:])
        self.write_progress()
        print(
            f"FIN : {len(self.done)}/{len(self.projects)} projets, "
            f"arrêt pour le temps = {self.stopped_for_time}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dossier contenant projects.jsonl")
    parser.add_argument("--out", default="/kaggle/working")
    parser.add_argument("--urls", nargs="+", default=["http://127.0.0.1:11434/api/chat"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--deadline-hours", type=float, default=11.0)
    parser.add_argument("--limit", type=int, default=None, help="pour un essai local")
    parser.add_argument("--resume-from", nargs="*", default=[], help="anciens results.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    projects = load_jsonl(Path(args.data) / "projects.jsonl")
    if args.limit:
        projects = sorted(projects, key=lambda p: p["order"])[: args.limit]
    done = set()
    for path in args.resume_from:  # résultats d'une session précédente : recopiés et sautés
        old = load_jsonl(Path(path))
        with (out / "results.jsonl").open("a", encoding="utf-8") as f:
            for r in old:
                if r.get("judge") is not None and r["key"] not in done:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done.add(r["key"])
    Runner(projects, out, args.urls, args.model, args.rounds,
           args.deadline_hours * 3600, done_keys=done).run_all()  # fmt: skip


if __name__ == "__main__":
    main()
