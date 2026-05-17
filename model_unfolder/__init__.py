"""model_unfolder — turn any HuggingFace transformer into a clear architecture diagram.

Quick start in a Jupyter notebook::

    from model_unfolder import unfold
    unfold("moonshotai/Kimi-K2-Instruct")

Outside Jupyter::

    diagram = unfold(cfg)
    diagram.save("kimi_k2.html")
"""
from .diagram import Diagram
from .evidence import inspect_model_code
from .parser import config_to_ir
from .ir import ModelIR, LayerSpec, AttentionSpec, FFNSpec, CrossLayerEdge
from .params import estimate_params

__version__ = "0.2.2"

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
    "inspect_model_code",
    "estimate_params",
]


def unfold(cfg_or_id, token=None, *, inspect_code=False, code_source="local") -> Diagram:
    """Unfold a transformer into a renderable architecture diagram.

    Parameters
    ----------
    cfg_or_id
        A HuggingFace ``PretrainedConfig`` instance, a model ID string
        (e.g. ``"moonshotai/Kimi-K2-Instruct"``), or a plain ``dict`` of
        ``config.json`` contents.
    token
        Optional Hugging Face token used only when ``cfg_or_id`` is a model ID.
        If omitted, ``HF_TOKEN`` and legacy Hugging Face token env vars are used
        when present.
    inspect_code
        If True, attach static source-code evidence to the IR. The code scanner
        parses modeling files as text/AST and does not execute model code.
    code_source
        Source for code inspection: ``"local"`` (installed transformers),
        ``"path"``, ``"hub"``, ``"auto"``, or a local file/directory path.

    Returns
    -------
    Diagram
        Renders inline in Jupyter; otherwise call ``.save()`` or ``.to_html()``.
    """
    ir = config_to_ir(
        cfg_or_id,
        token=token,
        inspect_code=inspect_code,
        code_source=code_source,
    )
    return Diagram(ir)


# friendly alias
show = unfold
