"""Test minimal : le paquet et ses sous-modules s'importent."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "investagents",
        "investagents.data",
        "investagents.schemas",
        "investagents.agents",
        "investagents.debate",
        "investagents.graph",
        "investagents.ml",
        "investagents.selection",
        "investagents.app",
    ],
)
def test_import(module: str) -> None:
    importlib.import_module(module)
