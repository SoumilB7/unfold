"""Transformer-LLM adapter.

There is exactly one parser (``parser.py``); see its module docstring for
the principle (config-driven, no per-family code).
"""
from . import parser

ADAPTERS = [parser]
