"""Temporary S8 comparison flag; never enabled by a product input/config."""
from contextlib import contextmanager
from contextvars import ContextVar


_legacy_comparison = ContextVar("unet_legacy_differential", default=False)


def legacy_comparison_enabled():
    return _legacy_comparison.get()


@contextmanager
def legacy_unet_comparison():
    """Verification-only old-path witness, scoped to the calling context."""
    token = _legacy_comparison.set(True)
    try:
        yield
    finally:
        _legacy_comparison.reset(token)
