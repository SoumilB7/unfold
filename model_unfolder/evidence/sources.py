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
    "openai-gpt": "openai",
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
        if local.files:
            return local
        # Diffusion DiT/UNet code lives in `diffusers`, not `transformers` — resolve
        # the modeling file by the config's `_class_name` (FluxTransformer2DModel, …).
        diff = _installed_diffusers_bundle(target)
        if diff is not None and diff.files:
            return diff
        if source == "local":
            return diff if (diff is not None and diff.warnings) else local

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
    return SourceBundle(
        source="path", files=files, model_id=str(path),
        warnings=() if files else ("No Python files found.",),
        component_files={"root": files} if files else {},
    )


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

    package_file = getattr(transformers, "__file__", None)
    if not package_file:
        return SourceBundle(
            source="local",
            model_type=model_type,
            architecture=architecture,
            model_id=model_id,
            warnings=(
                "transformers has no filesystem package path; cannot inspect "
                "local modeling source.",
            ),
        )
    models_root = Path(package_file).resolve().parent / "models"
    files: list[str] = []
    warnings: list[str] = []
    seen_files: set[str] = set()
    component_files: dict[str, tuple[str, ...]] = {}
    component_model_types: dict[str, str] = {}
    component_architectures: dict[str, str] = {}

    # Composite HF configs delegate real computation to nested component configs:
    # ``AutoModel.from_config(config.vision_config)`` and a separate text model are
    # common.  Looking up only the root wrapper makes the oracle appear present
    # while omitting the classes that actually perform the work.  Walk every
    # nested ``*_config`` structurally and gather its installed modeling source.
    for component, cfg in _component_configs(target):
        component_type = _own_model_type(cfg)
        if component == "root" and not component_type:
            component_type = model_type
        if not component_type:
            warnings.append(f"Could not infer model_type for Transformers component {component!r}.")
            continue
        component_model_types[component] = component_type
        component_architecture = (
            (_own_architecture(cfg) or _auto_model_architecture(component_type))
            if component == "root"
            else (_auto_model_architecture(component_type) or _own_architecture(cfg))
        )
        if component_architecture:
            component_architectures[component] = component_architecture
        family_dir = _transformers_family_dir(models_root, component_type)
        if family_dir is None:
            warnings.append(
                f"No installed Transformers source directory for component {component!r} "
                f"(model_type={component_type!r})."
            )
            continue
        modeling_files = tuple(sorted((models_root / family_dir).glob("modeling*.py")))
        if not modeling_files:
            warnings.append(
                f"No modeling*.py files found for component {component!r} "
                f"(model_type={component_type!r})."
            )
        component_paths = tuple(str(path) for path in modeling_files)
        if component_paths:
            component_files[component] = component_paths
        for path in component_paths:
            value = path
            if value not in seen_files:
                seen_files.add(value)
                files.append(value)

    return SourceBundle(
        source="local",
        files=tuple(files),
        model_type=model_type,
        architecture=architecture,
        model_id=model_id,
        warnings=tuple(warnings) if warnings else (() if files else (
            f"No modeling*.py files found for model_type={model_type!r}.",
        )),
        component_files=component_files,
        component_model_types=component_model_types,
        component_architectures=component_architectures,
    )


def _component_configs(target: Any):
    """Yield ``(qualified_path, config)`` for root and nested component configs.

    Only fields named ``*_config`` are traversed.  This follows Hugging Face's
    composite-config contract without mistaking arbitrary dictionaries (rope
    scaling, quantization settings, generation options) for model components.
    Object identity guards recursive/shared config objects.
    """
    seen: set[int] = set()

    def walk(value: Any, path: str):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        yield path, value
        items = value.items() if isinstance(value, dict) else vars(value).items() \
            if hasattr(value, "__dict__") else ()
        for name, child in items:
            if str(name).endswith("_config") and child is not None:
                child_path = str(name) if path == "root" else f"{path}.{name}"
                yield from walk(child, child_path)

    yield from walk(target, "root")


def _own_model_type(target: Any) -> str | None:
    """Return only this config object's model type, never a nested fallback."""
    value = _get_value(target, "model_type")
    return str(value) if value else None


def _own_architecture(target: Any) -> str | None:
    arches = _get_value(target, "architectures")
    if arches:
        try:
            return str(arches[0])
        except (TypeError, IndexError):
            return str(arches)
    return None


def _auto_model_architecture(model_type: str) -> str | None:
    """The installed Transformers AutoModel mapping, read without model import.

    Composite component configs commonly omit ``architectures``.  The static
    mapping is the authoritative config-type -> concrete model class relation
    used by ``AutoModel.from_config`` itself, and lets conformance start from the
    exact delegated model instead of every class sharing its source file.
    """
    try:
        from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    except (ImportError, AttributeError):
        return None
    value = MODEL_MAPPING_NAMES.get(model_type)
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    return str(value) if value else None


def _transformers_family_dir(models_root: Path, model_type: str) -> str | None:
    family_dir = MODEL_TYPE_TO_TRANSFORMERS_DIR.get(model_type)
    if family_dir is not None and (models_root / family_dir).exists():
        return family_dir
    return _direct_transformers_family_dir(models_root, model_type)


def _looks_like_diffusion_class(cls: str) -> bool:
    """Whether ``cls`` names a diffusion denoiser — by the GENERAL marker vocabulary
    (everchanging ``dit_class_markers`` + UNet), never a hand-picked substring. The
    old narrow ``"Transformer"/"UNet"`` gate missed ``HunyuanDiT2DModel`` /
    ``LuminaNextDiT2DModel`` (they carry "DiT", not "Transformer"), wrongly reporting
    their installed source as MISSING and silently skipping conformance + the
    code-derived FFN. Reuses the same markers the diffusor adapter detects on."""
    from ..everchanging import load_diffusion_typing
    markers = tuple(load_diffusion_typing().get("dit_class_markers") or ()) + ("UNet", "Transformer")
    return any(m in cls for m in markers)


def _installed_diffusers_bundle(target: Any) -> SourceBundle | None:
    """Resolve a diffusion model's modeling file in the installed ``diffusers``.

    Diffusion configs name their class in ``_class_name`` (e.g.
    ``FluxTransformer2DModel``); the file that defines it also defines the
    transformer block, whose norm/attention instantiations are what we read.
    Returns ``None`` when the target isn't a diffusion class or diffusers is
    absent (so the caller falls back to the transformers bundle)."""
    cls = _string_value(target, "_class_name")
    if not cls or not _looks_like_diffusion_class(cls):
        return None
    try:
        import diffusers
    except ImportError:
        return None
    import re
    models_root = Path(diffusers.__file__).resolve().parent / "models"
    if not models_root.exists():
        return None
    pat = re.compile(rf"^class {re.escape(cls)}\b", re.M)
    model_id = _model_id(target)
    for f in sorted(models_root.rglob("*.py")):
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if pat.search(text):
            return SourceBundle(source="local", files=(str(f),),
                                architecture=cls, model_id=model_id,
                                component_files={"root": (str(f),)},
                                component_architectures={"root": cls})
    return SourceBundle(
        source="local", architecture=cls, model_id=model_id,
        warnings=(f"No installed diffusers modeling file defines {cls!r}.",),
    )


def _direct_transformers_family_dir(models_root: Path, model_type: str) -> str | None:
    """Use the installed Transformers family directory when it matches directly.

    The explicit map above covers families whose model_type differs from the
    package directory (for example ``gpt_j`` -> ``gptj``). Many newer wrappers
    use the model_type as the directory name, so this keeps multimodal additions
    like qwen2_audio from needing one-off source-map entries.
    """
    normalized = model_type.replace("-", "_")
    candidates = [model_type, normalized]
    # Nested HF config types often describe the component role while sharing the
    # parent's implementation package: qwen3_5_text -> qwen3_5,
    # siglip_vision_model -> siglip.  Strip only recognized role suffixes and
    # accept the result solely when that installed family directory exists.
    for suffix in ("_vision_model", "_text_model", "_audio_model",
                   "_vision", "_text", "_audio"):
        if normalized.endswith(suffix):
            candidates.append(normalized[:-len(suffix)])
    for candidate in candidates:
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
        component_files={"root": files} if files else {},
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
