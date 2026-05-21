"""Source discovery for static Hugging Face modeling-code inspection."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import SourceBundle


MODEL_TYPE_TO_TRANSFORMERS_DIR = {
    # Text-only transformer LMs
    "baichuan": "baichuan",
    "bloom": "bloom",
    "dbrx": "dbrx",
    "deepseek_v2": "deepseek_v2",
    "deepseek_v3": "deepseek_v3",
    "falcon": "falcon",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gpt_bigcode": "gpt_bigcode",
    "gpt_j": "gptj",
    "gpt_neox": "gpt_neox",
    "gpt_oss": "gpt_oss",
    "llama": "llama",
    "llama4": "llama4",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "mpt": "mpt",
    "olmo": "olmo",
    "olmo2": "olmo2",
    "olmoe": "olmoe",
    "opt": "opt",
    "phi": "phi",
    "phi3": "phi3",
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2_moe",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3_moe",
    "stablelm": "stablelm",

    # Multi-modal wrappers (transformer LM with vision/audio encoders).
    # The wrapper directory contains both the LM and the modality pieces, so
    # the evidence scanner sees attention/FFN/MoE patterns alongside the
    # multimodal projector classes.
    "gemma3":           "gemma3",
    "gemma3_text":      "gemma3",
    "gemma3n":          "gemma3n",
    "gemma3n_text":     "gemma3n",
    "gemma4":           "gemma4",
    "gemma4_text":      "gemma4",
    "mllama":           "mllama",
    "llava":            "llava",
    "llava_next":       "llava_next",
    "llava_onevision":  "llava_onevision",
    "paligemma":        "paligemma",
    "qwen2_vl":         "qwen2_vl",
    "qwen2_5_vl":       "qwen2_5_vl",
    "qwen3_vl":         "qwen3_vl",
    "idefics2":         "idefics2",
    "idefics3":         "idefics3",
    "smolvlm":          "smolvlm",
    "internvl":         "internvl",
    "pixtral":          "pixtral",
    "fuyu":             "fuyu",
}


def resolve_source_files(target: Any, *, source: str = "local", token: Any = None) -> SourceBundle:
    """Resolve Python modeling files without executing model code.

    Parameters
    ----------
    target
        Config dict/object, model id, or local file/directory path.
    source
        ``"local"`` uses the installed ``transformers`` package, ``"path"``
        treats ``target`` as a file/directory, ``"hub"`` downloads only source
        files from the Hub, and ``"auto"`` tries local before Hub.
    """
    if source not in {"local", "path", "hub", "auto"}:
        raise ValueError("source must be one of: local, path, hub, auto")

    path_bundle = _path_bundle(target)
    if path_bundle is not None:
        return path_bundle

    if source in {"local", "auto"}:
        local = _installed_transformers_bundle(target)
        if local.files or source == "local":
            return local

    if source in {"hub", "auto"}:
        return _hub_bundle(target, token=token)

    return SourceBundle(
        source=source,
        model_type=_model_type(target),
        architecture=_architecture(target),
        model_id=_model_id(target),
        warnings=("No source files found.",),
    )


def _path_bundle(target: Any) -> SourceBundle | None:
    if not isinstance(target, (str, os.PathLike)):
        return None
    path = Path(target)
    if not path.exists():
        return None
    if path.is_file():
        files = (str(path),) if path.suffix == ".py" else ()
    else:
        files = tuple(str(p) for p in sorted(path.rglob("*.py")) if p.is_file())
    return SourceBundle(source="path", files=files, model_id=str(path), warnings=() if files else ("No Python files found.",))


def _installed_transformers_bundle(target: Any) -> SourceBundle:
    model_type = _model_type(target) or _guess_model_type_from_id(_model_id(target))
    architecture = _architecture(target)
    model_id = _model_id(target)
    if not model_type:
        return SourceBundle(
            source="local",
            model_type=model_type,
            architecture=architecture,
            model_id=model_id,
            warnings=("Could not infer model_type for installed Transformers source lookup.",),
        )

    try:
        import transformers
    except ImportError:
        return SourceBundle(
            source="local",
            model_type=model_type,
            architecture=architecture,
            model_id=model_id,
            warnings=("transformers is not installed; cannot inspect local modeling source.",),
        )

    family_dir = MODEL_TYPE_TO_TRANSFORMERS_DIR.get(model_type)
    models_root = Path(transformers.__file__).resolve().parent / "models"
    if family_dir is None:
        family_dir = _direct_transformers_family_dir(models_root, model_type)
    if family_dir is None:
        return SourceBundle(
            source="local",
            model_type=model_type,
            architecture=architecture,
            model_id=model_id,
            warnings=(
                f"No installed Transformers source directory for model_type={model_type!r}. "
                "Use code_source='hub' or pass a local modeling file/directory for code evidence.",
            ),
        )

    root = models_root / family_dir
    if not root.exists():
        return SourceBundle(
            source="local",
            model_type=model_type,
            architecture=architecture,
            model_id=model_id,
            warnings=(f"No installed Transformers source directory for model_type={model_type!r}.",),
        )
    files = tuple(str(p) for p in sorted(root.glob("modeling*.py")))
    return SourceBundle(
        source="local",
        files=files,
        model_type=model_type,
        architecture=architecture,
        model_id=model_id,
        warnings=() if files else (f"No modeling*.py files found for model_type={model_type!r}.",),
    )


def _direct_transformers_family_dir(models_root: Path, model_type: str) -> str | None:
    """Use the installed Transformers family directory when it matches directly.

    The explicit map above covers families whose model_type differs from the
    package directory (for example ``gpt_j`` -> ``gptj``). Many newer wrappers
    use the model_type as the directory name, so this keeps multimodal additions
    like qwen2_audio from needing one-off source-map entries.
    """
    for candidate in (model_type, model_type.replace("-", "_")):
        if candidate and (models_root / candidate).exists():
            return candidate
    return None


def _hub_bundle(target: Any, *, token: Any = None) -> SourceBundle:
    model_id = _model_id(target)
    if not model_id:
        return SourceBundle(source="hub", warnings=("Hub source lookup needs a model id.",))
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return SourceBundle(
            source="hub",
            model_type=_model_type(target),
            architecture=_architecture(target),
            model_id=model_id,
            warnings=("huggingface_hub is not installed; cannot download source files.",),
        )

    kwargs = {
        "repo_id": model_id,
        "allow_patterns": ["*.py", "**/*.py", "config.json"],
        "ignore_patterns": ["*.bin", "*.safetensors", "*.pt", "*.gguf", "*.onnx"],
    }
    clean_token = _clean_token(token)
    if clean_token is not None:
        kwargs["token"] = clean_token
    root = Path(snapshot_download(**kwargs))
    files = tuple(str(p) for p in sorted(root.rglob("*.py")) if p.is_file())
    return SourceBundle(
        source="hub",
        files=files,
        model_type=_model_type(target),
        architecture=_architecture(target),
        model_id=model_id,
        warnings=() if files else (f"No Python source files found in Hub repo {model_id!r}.",),
    )


def _model_type(target: Any) -> str | None:
    value = _get_value(target, "model_type")
    if value:
        return str(value)
    text_config = _get_value(target, "text_config")
    if text_config is not None:
        value = _get_value(text_config, "model_type")
        if value:
            return str(value)
    return None


def _architecture(target: Any) -> str | None:
    arches = _get_value(target, "architectures")
    if arches:
        try:
            return str(arches[0])
        except (TypeError, IndexError):
            return str(arches)
    return None


def _model_id(target: Any) -> str | None:
    if isinstance(target, str) and not Path(target).exists():
        return target
    return (
        _string_value(target, "_name_or_path")
        or _string_value(target, "name_or_path")
        or _string_value(target, "model_id")
        or _string_value(target, "repo_id")
    )


def _get_value(target: Any, name: str, default=None):
    if isinstance(target, dict):
        return target.get(name, default)
    return getattr(target, name, default)


def _string_value(target: Any, name: str) -> str | None:
    value = _get_value(target, name)
    return str(value) if value else None


def _clean_token(token: Any):
    if isinstance(token, str):
        token = token.strip()
        return token or None
    return token


def _guess_model_type_from_id(model_id: str | None) -> str | None:
    if not model_id:
        return None
    value = model_id.lower()
    checks = (
        ("deepseek-v3", "deepseek_v3"),
        ("deepseek-r1", "deepseek_v3"),
        ("deepseek-v2", "deepseek_v2"),
        ("qwen3-moe", "qwen3_moe"),
        ("qwen2-moe", "qwen2_moe"),
        ("qwen3", "qwen3"),
        ("qwen2", "qwen2"),
        ("mixtral", "mixtral"),
        ("mistral", "mistral"),
        ("llama-4", "llama4"),
        ("llama4", "llama4"),
        ("llama", "llama"),
        ("gemma-3n", "gemma3n"),
        ("gemma-3", "gemma3"),
        ("gemma-2", "gemma2"),
        ("gemma", "gemma"),
        ("phi-3", "phi3"),
        ("phi3", "phi3"),
        ("phi", "phi"),
        ("falcon", "falcon"),
        ("dbrx", "dbrx"),
        ("olmoe", "olmoe"),
        ("olmo-2", "olmo2"),
        ("olmo2", "olmo2"),
        ("olmo", "olmo"),
        ("gpt-oss", "gpt_oss"),
        ("gpt-neox", "gpt_neox"),
        ("gpt-j", "gpt_j"),
        ("bloom", "bloom"),
        ("opt-", "opt"),
        ("mpt", "mpt"),
    )
    for needle, model_type in checks:
        if needle in value:
            return model_type
    return None
