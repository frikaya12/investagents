"""Appel à un LLM local (Ollama) avec sortie JSON validée par Pydantic et 1 nouvel essai.

API locale d'Ollama (`POST http://127.0.0.1:11434/api/chat`, vérifiée sur la version 0.40.16) :
`format` reçoit le schéma JSON du modèle Pydantic, ce qui contraint la sortie. On valide quand
même avec Pydantic (règle : toute sortie invalide est régénérée une fois, puis signalée).
Coût : 0 € (modèle exécuté sur le PC).
"""

from __future__ import annotations

import time

import requests
from pydantic import BaseModel, ValidationError

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"


class LLMOutputError(RuntimeError):
    """Sortie toujours invalide après le nouvel essai."""


def call_json(
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    num_predict: int = 1500,
    num_ctx: int = 8192,
    seed: int = 0,
    post=requests.post,
) -> tuple[BaseModel, dict]:
    """Renvoie (objet validé, statistiques) ; lève LLMOutputError après 2 sorties invalides."""
    stats = {"attempts": 0, "seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0, "errors": []}
    for attempt in range(2):
        stats["attempts"] = attempt + 1
        t0 = time.time()
        r = post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": schema.model_json_schema(),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": num_ctx,
                    "seed": seed + attempt,
                },
            },
            timeout=3600,
        )
        r.raise_for_status()
        body = r.json()
        stats["seconds"] += time.time() - t0
        stats["prompt_tokens"] += body.get("prompt_eval_count", 0)
        stats["output_tokens"] += body.get("eval_count", 0)
        try:
            return schema.model_validate_json(body["message"]["content"]), stats
        except ValidationError as exc:
            stats["errors"].append(str(exc)[:500])
    raise LLMOutputError(f"Sortie invalide après 2 essais : {stats['errors'][-1]}")
