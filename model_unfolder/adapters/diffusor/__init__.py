"""Diffusor-family adapters.

Diffusion *transformers* (DiT/MMDiT — Flux, SD3, PixArt) get their own adapter
namespace here, but flow through the *same* IR -> Diagram -> renderer pipeline as
the transformer adapter: the partition is in detection + the model-level pipeline
skeleton, not a duplicate renderer.  See ``parser.py``.
"""
from . import parser

ADAPTERS = [parser]

