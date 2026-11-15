"""Client HTTP commun aux trois sources de données.

Une seule fonction, `get_json`, qui fait un GET et renvoie le JSON. Elle réessaie quand le serveur
répond « trop de requêtes » (429) ou une erreur temporaire (5xx, coupure réseau), en attendant de
plus en plus longtemps. Les API de la Banque mondiale limitent le débit : sans ça, un téléchargement
de plusieurs pages échoue au milieu.
"""

from __future__ import annotations

import time
from typing import Any

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    retries: int = 5,
    backoff_s: float = 5.0,
    timeout_s: float = 120.0,
    session: requests.Session | None = None,
    sleep=time.sleep,
) -> Any:
    """GET `url` et renvoie le JSON décodé ; réessaie sur 429 / 5xx / erreur réseau.

    `sleep` est injectable pour que les tests ne patientent pas vraiment.
    """
    http = session or requests
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = http.get(url, params=params, timeout=timeout_s)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
        else:
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
                return resp.json()
            last_error = requests.HTTPError(f"HTTP {resp.status_code} pour {resp.url}")
        if attempt < retries:
            sleep(backoff_s * (2**attempt))
    raise RuntimeError(f"Échec après {retries + 1} tentatives : {url}") from last_error
