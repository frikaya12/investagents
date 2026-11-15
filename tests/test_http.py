"""Tests du client HTTP : réessais sur 429 et erreurs réseau, sans vrai réseau."""

import pytest
import requests

from investagents.data.http import get_json


class FakeResponse:
    def __init__(self, status: int, payload=None):
        self.status_code = status
        self._payload = payload
        self.url = "http://test"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Renvoie une suite de réponses (ou d'exceptions) préparée à l'avance."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_returns_json_on_success():
    s = FakeSession([FakeResponse(200, {"ok": 1})])
    assert get_json("u", session=s, sleep=lambda _: None) == {"ok": 1}


def test_retries_on_429_then_succeeds():
    s = FakeSession([FakeResponse(429), FakeResponse(503), FakeResponse(200, [1])])
    waits = []
    assert get_json("u", session=s, sleep=waits.append, backoff_s=1) == [1]
    assert s.calls == 3
    assert waits == [1, 2]  # attente exponentielle


def test_retries_on_connection_error():
    s = FakeSession([requests.ConnectionError(), FakeResponse(200, "x")])
    assert get_json("u", session=s, sleep=lambda _: None) == "x"


def test_gives_up_after_retries():
    s = FakeSession([FakeResponse(429)] * 3)
    with pytest.raises(RuntimeError):
        get_json("u", session=s, retries=2, sleep=lambda _: None)


def test_does_not_retry_on_404():
    s = FakeSession([FakeResponse(404)])
    with pytest.raises(requests.HTTPError):
        get_json("u", session=s, sleep=lambda _: None)
    assert s.calls == 1
