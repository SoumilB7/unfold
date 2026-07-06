"""Shared test wiring.

CORPUS_DIR is the single source of the blessed-corpus location for tests.
It derives from ``model_unfolder.sable.DEFAULT_CORPUS`` so a corpus move is
a one-line change in sable.py — tests must never spell the folder name
themselves (it has moved once already and red-suited three tests).
"""
from pathlib import Path

import pytest

from model_unfolder.sable import DEFAULT_CORPUS as CORPUS_DIR


@pytest.fixture
def corpus_dir() -> Path:
    return CORPUS_DIR
