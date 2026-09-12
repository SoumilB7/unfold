"""Typed address/display markers for lawful identity use (plan §16.2).

Identity (a model's ``model_type`` / ``architectures`` / class name) may be used
for exactly two lawful purposes under I-1:

* **address** — to LOCATE the modeling source to read (which file, which
  installed config class). It never becomes a fact.
* **display** — as a human LABEL on a card. It never decides structure.

Previously the identity guard exempted these uses by hard-coding a set of
*function names* (``_ADDRESS_OR_DISPLAY_FUNCTIONS``).  That is exactly the
"exemption based solely on a name" §16 forbids: a new function is silently
auto-exempt, and a genuinely structural identity branch hiding in an
address-named function is invisible.

These decorators replace that with a **typed marker the author applies at the
site**.  A function that lawfully branches on identity for address/display
purposes must say so explicitly by wrapping itself; the guard recognizes the
decorator, not the name.  An undecorated function that branches on identity is
debt — the default is suspicion, and the exemption is a conscious, typed act.

The decorators are inert at runtime (identity return value unchanged); their
whole purpose is to carry the lawful ROLE into the static guard's AST scan and
to document intent at the call site.
"""
from __future__ import annotations

from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable)

#: Decorator names the identity guard recognizes as lawful identity roles.  The
#: guard reads these off a function's decorator list; it does not know function
#: names.  Keep this the single source of truth shared with the guard.
IDENTITY_ROLE_DECORATORS = frozenset({"identity_address", "identity_display"})


def identity_address(func: _F) -> _F:
    """Mark: this function branches on identity only to LOCATE source.

    The identity read decides *which* modeling file / installed config class to
    read — an address — never a structural fact.  The guard exempts identity
    branches inside a function carrying this marker; without it, such a branch
    is reported as debt.
    """
    return func


def identity_display(func: _F) -> _F:
    """Mark: this function branches on identity only to produce a human LABEL.

    The identity read decides *what to show* on a card — display provenance —
    never a structural fact (I-1 allows displaying identity).
    """
    return func


__all__ = ["IDENTITY_ROLE_DECORATORS", "identity_address", "identity_display"]
