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
from .lint import lint_labels
from .parser import config_to_ir
from .sable import SableReport, bless, check_regression, load_corpus, sable
from .ir import ModelIR, LayerSpec, AttentionSpec, FFNSpec, CrossLayerEdge
from .params import estimate_params
from .errors import (
    UnfoldError,
    ModelNotFoundError,
    ModelAccessError,
    ConfigParseError,
)

__version__ = "0.2.15"

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
    "sable",
    "SableReport",
    "bless",
    "check_regression",
    "load_corpus",
    "lint_labels",
    "UnfoldError",
    "ModelNotFoundError",
    "ModelAccessError",
    "ConfigParseError",
]


def unfold(
    cfg_or_id,
    token=None,
    *,
    inspect_code: bool = False,
    code_source: str = "local",
    return_json: bool = False,
):
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
    return_json
        If True, return the expanded architecture JSON dict instead of the
        renderable ``Diagram``.  The JSON uses stable structural fields for
        dimensions, projections, layer groups, operation graphs, cache behavior,
        and trace paths instead of renderer labels/descriptions.

    Returns
    -------
    Diagram | dict
        ``Diagram`` by default; ``dict`` when ``return_json=True``.
    """
    ir = config_to_ir(
        cfg_or_id,
        token=token,
        inspect_code=inspect_code,
        code_source=code_source,
    )
    diagram = Diagram(ir)
    return diagram.to_json() if return_json else diagram


# friendly alias
show = unfold
