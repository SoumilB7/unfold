"""Compatibility wrapper for the HTML/SVG renderer backend."""

from .renderers.html import render_document, render_fragment

__all__ = ["render_document", "render_fragment"]
