"""Worker-only exact framework-type witness, captured before model imports."""
from dataclasses import dataclass
import hashlib
import inspect


@dataclass(frozen=True)
class FrameworkPrimitiveWitness:
    key: str
    forward_source_sha256: str

    def __post_init__(self):
        if self.key not in {"linear", "conv1d", "conv2d", "conv3d", "group_norm",
                            "layer_norm", "rms_norm", "silu", "gelu", "relu", "dropout"}:
            raise ValueError("framework primitive witness has a closed key")
        if len(self.forward_source_sha256) != 64:
            raise ValueError("framework method witness requires a source hash")


def capture_framework_types():
    """Return live object handles; no model/class string participates in membership."""
    import torch.nn as nn
    types = {nn.Linear: "linear", nn.Conv1d: "conv1d", nn.Conv2d: "conv2d",
             nn.Conv3d: "conv3d", nn.GroupNorm: "group_norm", nn.LayerNorm: "layer_norm",
             nn.RMSNorm: "rms_norm", nn.SiLU: "silu", nn.GELU: "gelu",
             nn.ReLU: "relu", nn.Dropout: "dropout"}
    return {cls: (key, cls.forward, cls.forward.__code__, cls.__call__,
                  hashlib.sha256(inspect.getsource(cls.forward).encode()).hexdigest())
            for cls, key in types.items()}


def witness_framework_type(module, captured):
    row = captured.get(type(module))
    if row is None:
        return None
    key, forward, code, call, digest = row
    cls = type(module)
    if cls.forward is not forward or cls.forward.__code__ is not code or cls.__call__ is not call:
        return None
    if "forward" in vars(module) or module._forward_hooks or module._forward_pre_hooks:
        return None
    return FrameworkPrimitiveWitness(key, digest)
