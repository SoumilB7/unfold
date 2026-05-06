"""unfold — turn any HuggingFace transformer into a clear architecture diagram.

Quick start in a Jupyter notebook::

    from unfold import unfold
    unfold("moonshotai/Kimi-K2-Instruct")

Outside Jupyter::

    diagram = unfold(cfg)
    diagram.save("kimi_k2.html")
"""
from .diagram import Diagram
from .parser import config_to_ir
from .ir import ModelIR, LayerSpec, AttentionSpec, FFNSpec, CrossLayerEdge
from .params import estimate_params

__version__ = "0.2.0"

__all__ = [
    "unfold",
    "show",
    "Diagram",
    "ModelIR",
    "LayerSpec",
    "AttentionSpec",
    "FFNSpec",
    "CrossLayerEdge",
    "config_to_ir",
    "estimate_params",
]


def unfold(cfg_or_id) -> Diagram:
    """Unfold a transformer into a renderable architecture diagram.

    Parameters
    ----------
    cfg_or_id
        A HuggingFace ``PretrainedConfig`` instance, a model ID string
        (e.g. ``"moonshotai/Kimi-K2-Instruct"``), or a plain ``dict`` of
        ``config.json`` contents.

    Returns
    -------
    Diagram
        Renders inline in Jupyter; otherwise call ``.save()`` or ``.to_html()``.
    """
    ir = config_to_ir(cfg_or_id)
    return Diagram(ir)


# friendly alias
show = unfold
