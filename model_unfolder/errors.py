"""Public exception types raised by model-unfolder.

All inherit from :class:`UnfoldError`, so a caller can ``except UnfoldError`` to
catch everything.  The three concrete types separate the failure modes users
actually need to tell apart and act on:

* :class:`ModelNotFoundError` — the id is wrong, or the architecture isn't known
  to the installed ``transformers`` (often a stale install).
* :class:`ModelAccessError` — the model exists but is gated/private (token).
* :class:`ConfigParseError` — a config loaded but can't become a usable diagram.

When the debug switch is on (``MODEL_UNFOLDER_DEBUG=1``) every one of these is
also printed with its underlying cause before being raised; see
:mod:`model_unfolder.adapters.transformer.debug`.
"""
from __future__ import annotations


class UnfoldError(Exception):
    """Base class for every error raised by model-unfolder."""


class ModelNotFoundError(UnfoldError):
    """The model id couldn't be located, or its architecture isn't recognized.

    A common cause is a stale ``transformers`` install — a newly released
    architecture may not exist in the version you have, so updating it
    (``pip install -U transformers``) often fixes it.
    """


class ModelAccessError(UnfoldError):
    """The model exists but couldn't be accessed — gated/private/auth (401/403).

    The fix is almost always passing a Hugging Face token that has access
    (``token=...`` or the ``HF_TOKEN`` env var) and accepting the model's
    license on its Hugging Face page.
    """


class ConfigParseError(UnfoldError):
    """A config was loaded but couldn't be turned into a usable diagram.

    Distinct from the soft "partial config" warning: this means the result is
    fundamentally broken (e.g. no transformer layers at all), not merely
    missing some annotations.
    """


__all__ = [
    "UnfoldError",
    "ModelNotFoundError",
    "ModelAccessError",
    "ConfigParseError",
]
