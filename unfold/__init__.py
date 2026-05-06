"""transformer_viz — auto-generated architecture diagrams for HF transformer models.

Quick start in a Jupyter notebook::

    from transformer_viz import visualize
    visualize("moonshotai/Kimi-K2-Instruct")

Outside Jupyter::

    diagram = visualize(cfg)
    diagram.save("kimi_k2.html")
"""
from .diagram import TransformerDiagram
from .parser import config_to_ir
from .ir import ModelIR, LayerSpec, AttentionSpec, FFNSpec, CrossLayerEdge

__version__ = "0.1.0"

__all__ = [
    "visualize",
    "TransformerDiagram",
    "ModelIR",
    "LayerSpec",
    "AttentionSpec",
    "FFNSpec",
    "CrossLayerEdge",
    "config_to_ir",
]


def visualize(cfg_or_id) -> TransformerDiagram:
    """Render a transformer architecture diagram.

    Parameters
    ----------
    cfg_or_id
        A HuggingFace ``PretrainedConfig`` instance, a model ID string
        (e.g. ``"moonshotai/Kimi-K2-Instruct"``), or a plain ``dict`` of
        ``config.json`` contents.

    Returns
    -------
    TransformerDiagram
        Renders inline in Jupyter; otherwise call ``.save()`` or ``.to_html()``.
    """
    ir = config_to_ir(cfg_or_id)
    return TransformerDiagram(ir)
