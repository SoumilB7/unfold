"""User-editable config vocabulary — *data, not code*.

Files here are loaded at runtime so new config dialects can be supported by
editing YAML, with the parser/renderer code left untouched.  Anything that is
config-variable vocabulary — aliases, ignore lists, word types, approved block
typing — lives here as YAML, never hardcoded.

Organised by **domain** (one folder per adapter family), so it scales as new
families are added — drop a folder + YAML, no code change beyond a thin loader::

    everchanging/
      transformer/
        aliases.yaml          canonical field -> config key spellings (in order)
        ignored_fields.yaml   non-architectural keys/suffixes (unparsed diagnostic)
      diffusor/
        aliases.yaml          DiT/MMDiT field aliases
        typing.yaml           APPROVED block stages / ids / DiT detection markers
        text_encoders.yaml    text-encoder class name -> friendly label

Loading prefers PyYAML when installed; otherwise a tiny built-in reader handles
the flow-style (``key: [a, b, c]``) and block-style (``key:`` + ``- item``)
formats the shipped files use, so the package keeps working with no third-party
dependency.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load(domain: str, name: str) -> dict:
    """Load ``everchanging/<domain>/<name>.yaml`` (e.g. ``load("transformer", "aliases")``)."""
    text = (_DIR / domain / f"{name}.yaml").read_text(encoding="utf-8")
    try:
        import yaml  # optional; not a hard dependency
    except ImportError:
        return _parse_flow_yaml(text)
    return yaml.safe_load(text) or {}


# --- transformer domain -----------------------------------------------------

def load_aliases() -> dict[str, list[str]]:
    """Global transformer field aliases.

    Scope-qualified input-format entries share ``transformer/aliases.yaml`` so
    all spelling equivalence has one vocabulary, but they are deliberately
    excluded here: a dialect-local alias such as ``dim`` must never leak into
    the unordered transformer config scope.
    """
    return {
        key: value
        for key, value in load("transformer", "aliases").items()
        if not str(key).startswith("input_format.")
    }


def load_input_format_aliases(format_name: str) -> dict[str, list[str]]:
    """Return scope-qualified aliases for one structurally detected format.

    Keys are stored as ``input_format.<format>.<scope>.<canonical>`` in the
    shared alias resource.  ``format_name`` is a syntax contract, never a model,
    class, or repository identity.
    """
    prefix = f"input_format.{format_name}."
    return {
        str(key)[len(prefix):]: list(value)
        for key, value in load("transformer", "aliases").items()
        if str(key).startswith(prefix) and isinstance(value, list)
    }


def load_ignored_fields() -> dict[str, list[str]]:
    """Non-architectural config keys/suffixes (``transformer/ignored_fields.yaml``)."""
    data = load("transformer", "ignored_fields")
    return {
        "keys": data.get("keys") or [],
        "suffixes": data.get("suffixes") or [],
        "opaque_scopes": data.get("opaque_scopes") or [],
    }


def load_ledger_ignores() -> dict:
    """The ledger-side scoped-ignore vocabulary (evidence/ledger_ignores.yaml).

    Soumil's final vet (2026-07-19): ignores are EXACT SCOPED RULES — each
    rule row is ``"<owner pattern> | <exact path> | <reason>"`` — plus the
    address-key list, whose entries match only a top-level read of the key
    itself.  The transformer ``ignored_fields`` diagnostic vocabulary no
    longer feeds the ledger: quieting an unread-fields REPORT and classifying
    a READ as non-architectural are different powers, and the global feed is
    how an architectural field (is_encoder_decoder) got recorded as plumbing."""
    data = load("evidence", "ledger_ignores")
    rules = []
    for row in data.get("rules") or []:
        parts = [p.strip() for p in str(row).split("|", 2)]
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"ledger_ignores rule must be '<owner> | <path> | <reason>', "
                f"got {row!r}")
        rules.append(tuple(parts))
    return {"address_keys": data.get("address_keys") or [], "rules": rules}


def load_transformer_typing() -> dict[str, list[str]]:
    """APPROVED transformer block stages (``transformer/typing.yaml``) — the known
    decoder-only-transformer block taxonomy."""
    data = load("transformer", "typing")
    return {"stages": data.get("stages") or []}


def load_decoderness() -> dict[str, list[str]]:
    """Config-declared decoder-ness vocabulary (``transformer/decoderness.yaml``):
    the ``architectures[]`` class-role suffixes that declare an autoregressive
    decoder (causal-LM heads; decoder-only generation wrappers, honored only
    without ``is_encoder_decoder``). U2 mask default-kill."""
    data = load("transformer", "decoderness")
    return {
        "causal_lm_suffixes": data.get("causal_lm_suffixes") or [],
        "wrapper_generation_suffixes": data.get("wrapper_generation_suffixes") or [],
    }


def load_composite_slots() -> dict:
    """Declared component slots of composite/seq2seq configs
    (``transformer/composite_slots.yaml``).

    Returns ``{"slots": {slot_key: role}, "cross_attn_fields": [name, ...]}``
    where role is ``main`` / ``encoder`` / ``codec``.  A slot only counts as a
    component when its config value actually declares a ``model_type`` —
    callers must apply that evidence gate; this is just the name vocabulary."""
    data = load("transformer", "composite_slots")
    slots: dict[str, str] = {}
    for item in data.get("slots") or []:
        if isinstance(item, str) and "=" in item:
            key, _, role = item.partition("=")
            slots[key.strip()] = role.strip()
    undrawn: dict[str, str] = {}
    for item in data.get("undrawn_component_fields") or []:
        if isinstance(item, str) and "=" in item:
            field, _, label = item.partition("=")
            undrawn[field.strip()] = label.strip()
    return {"slots": slots,
            "cross_attn_fields": list(data.get("cross_attn_fields") or []),
            "undrawn_component_fields": undrawn}


# --- diffusor domain --------------------------------------------------------

def load_diffusion_aliases() -> dict[str, list[str]]:
    """Diffusion (DiT/MMDiT) field-alias table (``diffusor/aliases.yaml``)."""
    return load("diffusor", "aliases")


def load_diffusion_config_facts() -> dict[str, list[dict]]:
    """Declared config FACT fields (``diffusor/config_facts.yaml``): per-stage
    buckets of ``{field, label, noop?, silent?}`` rows.  Every present field is
    READ (ownership audit) and non-noop, non-silent values become card chips."""
    data = load("diffusor", "config_facts")
    return {bucket: [row for row in (rows or []) if isinstance(row, dict) and row.get("field")]
            for bucket, rows in (data or {}).items()}


def load_diffusion_typing() -> dict[str, list[str]]:
    """APPROVED diffusion block typing (``diffusor/typing.yaml``): ``stages``
    (blessed diffusion_stage values), ``block_ids`` (reused transformer-layer
    ids), ``part_kinds`` (compound detail-view regions),
    ``dit_class_markers`` (detection substrings), and ``scheduler_display``
    ("Class=Display" name overrides)."""
    data = load("diffusor", "typing")
    return {
        "stages": data.get("stages") or [],
        "block_ids": data.get("block_ids") or [],
        "part_kinds": data.get("part_kinds") or [],
        "dit_class_markers": data.get("dit_class_markers") or [],
        "scheduler_display": data.get("scheduler_display") or [],
        "scheduler_flow_matching_markers": data.get("scheduler_flow_matching_markers") or [],
        "norm_type_kind": data.get("norm_type_kind") or [],
        "temporal_forward_markers": data.get("temporal_forward_markers") or [],
        "stack_lane_params": data.get("stack_lane_params") or [],
        "temporal_config_fields": data.get("temporal_config_fields") or [],
        "companion_denoiser_fields": data.get("companion_denoiser_fields") or [],
        "audio_vae_fields": data.get("audio_vae_fields") or [],
    }


def load_diffusion_text_encoders() -> dict[str, str]:
    """Text-encoder class name -> friendly label (``diffusor/text_encoders.yaml``;
    the whole file is the flat map)."""
    return {k: v for k, v in load("diffusor", "text_encoders").items() if isinstance(v, str)}


def load_diffusion_conditioning() -> dict[str, dict]:
    """Conditioning-MODALITY vocabulary (``diffusor/conditioning.yaml``): the
    declared ``encoder_hid_dim_type`` / ``addition_embed_type`` enum values mapped
    to their {modality, kv_label / add_label, projector} story.  An unmapped value
    falls to honest-unknown, never to text."""
    data = load("diffusor", "conditioning")
    return {
        "encoder_hid_dim_type": data.get("encoder_hid_dim_type") or {},
        "addition_embed_type": data.get("addition_embed_type") or {},
    }


# --- conformance domain (op-conformance diff: diagram structure vs HF forward) ---

def load_conformance_op_tokens() -> dict[str, str]:
    """``conformance/op_tokens.yaml`` inverted to ``{call_token: canonical_op}``.

    The file is authored canonical->[tokens] for readability; the extractor wants
    token->canonical. ``ignore`` maps its tokens to the sentinel ``""``."""
    out: dict[str, str] = {}
    for canonical, tokens in load("conformance", "op_tokens").items():
        for tok in tokens or []:
            out[str(tok)] = "" if canonical == "ignore" else str(canonical)
    return out


def load_conformance_type_roles() -> tuple[list[str], dict[str, list[str]]]:
    """``conformance/type_roles.yaml`` -> (priority_order, {role: [substrings]}).

    The extractor matches a constructed-class name against substrings in
    ``priority`` order (first hit wins), so a specific role beats generic
    ``linear``."""
    data = load("conformance", "type_roles")
    priority = [str(r) for r in (data.get("priority") or [])]
    roles = {k: [str(s) for s in (v or [])]
             for k, v in data.items() if k != "priority"}
    if not priority:                       # fall back to declared order
        priority = list(roles.keys())
    return priority, roles


def load_conformance_fact_markers() -> dict[str, list[str]]:
    """``conformance/fact_markers.yaml`` -> ``{marker: [substrings]}`` for the
    fact-conformance net (the SAME-op-kind, different-SEMANTICS dimensions
    op-presence is blind to): ``rotary`` (block applies RoPE -> a NoPE claim is
    fabricated) and ``linear_attn`` (self-attn is linear, not softmax)."""
    data = load("conformance", "fact_markers")
    return {k: [str(s) for s in (v or [])] for k, v in data.items()}


def load_conformance_map() -> dict:
    """``conformance/conformance_map.yaml`` -> view<->code overrides + markers.

    Returns ``{"views": {"<family>/<view>": "Class.method"},
    "single_stream_class_markers": [...]}``."""
    data = load("conformance", "conformance_map")
    views: dict[str, str] = {}
    for entry in data.get("views") or []:
        if isinstance(entry, str) and "=" in entry:
            key, _, val = entry.partition("=")
            views[key.strip()] = val.strip()
    return {
        "views": views,
        "single_stream_class_markers": [str(m) for m in (data.get("single_stream_class_markers") or [])],
    }


def load_conformance_abstractions() -> dict:
    """``conformance/abstractions.yaml`` -> the deliberate-abstraction allow-list.

    Returns ``{"omit_global": set, "omit_scoped": {"<family>/<view>": set(ops)},
    "composite": {drawn_op: set(absorbed_code_ops)}, "draw_extra":
    {"<family>/<view>": set(ops)}, "since": {citation_key: set(tokens)}}``."""
    data = load("conformance", "abstractions")

    def _scoped(entries):                  # ["family/view=op[,op…]", ...] -> {key: {ops}}
        out: dict[str, set[str]] = {}
        for e in entries or []:
            if isinstance(e, str) and "=" in e:
                key, _, ops = e.partition("=")
                out.setdefault(key.strip(), set()).update(
                    o.strip() for o in ops.split(",") if o.strip())
        return out

    composite: dict[str, set[str]] = {}
    for e in data.get("composite") or []:  # ["ffn=linear,activation", ...]
        if isinstance(e, str) and "=" in e:
            drawn, _, absorbed = e.partition("=")
            composite[drawn.strip()] = {a.strip() for a in absorbed.split(",") if a.strip()}
    since: dict[str, set[str]] = {}
    for e in data.get("since") or []:      # ["family/view=tokA,tokB", ...]
        if isinstance(e, str) and "=" in e:
            key, _, toks = e.partition("=")
            since[key.strip()] = {t.strip() for t in toks.split(",") if t.strip()}
    return {
        "omit_global": {str(o) for o in (data.get("omit_global") or [])},
        "omit_scoped": _scoped(data.get("omit_scoped")),
        "composite": composite,
        "draw_extra": _scoped(data.get("draw_extra")),
        "since": since,
    }


def load_conformance_wiring_roles() -> tuple[dict[str, str], dict[str, list[str]]]:
    """``conformance/wiring_roles.yaml`` -> (stage_role, role_params).

    ``stage_role``: a side-input's ``diffusion_stage`` -> conditioning role.
    ``role_params``: role -> forward()-parameter-name substrings that satisfy it.
    Both authored as ``"key=val[,val…]"`` flow entries so the loader needs no PyYAML."""
    data = load("conformance", "wiring_roles")
    stage_role: dict[str, str] = {}
    for e in data.get("stage_role") or []:
        if isinstance(e, str) and "=" in e:
            k, _, v = e.partition("=")
            stage_role[k.strip()] = v.strip()
    role_params: dict[str, list[str]] = {}
    for e in data.get("role_params") or []:
        if isinstance(e, str) and "=" in e:
            role, _, subs = e.partition("=")
            role_params[role.strip()] = [s.strip().lower() for s in subs.split(",") if s.strip()]
    return stage_role, role_params


def load_conformance_transitive() -> dict:
    """``conformance/transitive.yaml`` -> the recursive drill-conformance vocab.

    Returns a dict with: ``attention_compute_ops`` (frozenset), ``attention_compute_tokens``
    (frozenset), ``drawn_ignore`` (frozenset of drawn node-kinds dropped from the
    op diff), ``drawn_op_map`` (drawn-kind -> code-op), ``semantic_kinds`` (drawn
    kinds checked by marker presence, not op-kind), ``semantic_markers``
    ({kind: [substrings]}), and ``library_helpers`` (token -> frozenset(ops))."""
    data = load("conformance", "transitive")

    def _list(key):
        return [str(x) for x in (data.get(key) or [])]

    drawn_op_map: dict[str, str] = {}
    for e in _list("drawn_op_map"):            # ["select=route", ...]
        if "=" in e:
            k, _, v = e.partition("=")
            drawn_op_map[k.strip()] = v.strip()
    helpers: dict[str, frozenset[str]] = {}
    for e in _list("library_helpers"):         # ["repeat_kv=reshape", "dropout="]
        if "=" in e:
            name, _, ops = e.partition("=")
            helpers[name.strip()] = frozenset(o.strip() for o in ops.split(",") if o.strip())

    def _kv_list(key):                         # ["role=a,b", ...] -> {role: [a, b]}
        out: dict[str, list[str]] = {}
        for e in _list(key):
            if "=" in e:
                k, _, v = e.partition("=")
                out[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]
        return out

    def _kv_str(key):                          # ["role=type", ...] -> {role: type}
        return {k: (v[0] if v else "") for k, v in _kv_list(key).items()}

    return {
        "attention_compute_ops": frozenset(_list("attention_compute_ops")),
        "attention_compute_tokens": frozenset(_list("attention_compute_tokens")),
        "drawn_ignore": frozenset(_list("drawn_ignore")),
        "drawn_op_map": drawn_op_map,
        "semantic_kinds": frozenset(_list("semantic_kinds")),
        "semantic_markers": {
            "rope": [s.lower() for s in _list("semantic_rope_markers")],
            "cache": [s.lower() for s in _list("semantic_cache_markers")],
            "relative_bias": [s.lower() for s in _list("semantic_relative_bias_markers")],
        },
        "score_scale_markers": frozenset(s.lower() for s in _list("score_scale_markers")),
        # U2 P2d mask reader: mask-machinery call tokens, split by direction.
        "causal_mask_call_tokens": frozenset(_list("causal_mask_call_tokens")),
        "bidirectional_mask_call_tokens": frozenset(_list("bidirectional_mask_call_tokens")),
        "library_helpers": helpers,
        "drill_role_markers": _kv_list("drill_role_markers"),
        "component_view_markers": _kv_list("component_view_markers"),
        "component_class_markers": _kv_list("component_class_markers"),
        "role_field_markers": _kv_list("role_field_markers"),
        "drill_class_markers": _kv_list("drill_class_markers"),
        "drill_role_to_type": _kv_str("drill_role_to_type"),
        "drill_category": _kv_str("drill_category"),
        "drill_salient_missing": {k: frozenset(v) for k, v in _kv_list("drill_salient_missing").items()},
        "drill_op_equivalents": {k: frozenset(v) for k, v in _kv_list("drill_op_equivalents").items()},
        "selection_presentation_kinds": frozenset(_list("selection_presentation_kinds")),
        "composite_container_map": _kv_str("composite_container_map"),
        "processor_markers": frozenset(_list("processor_markers")),
        "constructor_classmethods": frozenset(_list("constructor_classmethods")),
    }


_CONSTRUCTOR_CLASSMETHODS: frozenset[str] | None = None


def load_constructor_classmethods() -> frozenset[str]:
    """Factory classmethods that construct their base class (``X._from_config``).

    Cached at module level: the shared AST extractor consults this set for every
    ``Attribute`` call it classifies, so the YAML must be read once, not per node.
    """
    global _CONSTRUCTOR_CLASSMETHODS
    if _CONSTRUCTOR_CLASSMETHODS is None:
        data = load("conformance", "transitive")
        _CONSTRUCTOR_CLASSMETHODS = frozenset(
            str(x) for x in (data.get("constructor_classmethods") or [])
        )
    return _CONSTRUCTOR_CLASSMETHODS


def load_program_index_vocab() -> dict[str, frozenset[str]]:
    """Syntactic vocabulary for the U3 ProgramIndex walker
    (``evidence/program_index_vocab.yaml``): the bare-Name config roots, the
    activation dispatch tables (``ACT2FN[…]``), the activation call forms
    (``get_activation(…)``), and the container constructor classes (ModuleList/
    Sequential/ModuleDict).  Data, never hardcoded — a new config dialect or
    container idiom is a YAML edit."""
    data = load("evidence", "program_index_vocab")
    return {
        "config_roots": frozenset(str(x) for x in (data.get("config_roots") or [])),
        "activation_dispatch": frozenset(
            str(x) for x in (data.get("activation_dispatch") or [])),
        "activation_calls": frozenset(
            str(x) for x in (data.get("activation_calls") or [])),
        "container_classes": frozenset(
            str(x) for x in (data.get("container_classes") or [])),
    }


def _parse_flow_yaml(text: str) -> dict[str, list[str]]:
    """Minimal reader for ``key: [a, b, c]`` / ``key:`` + ``- item`` blocks.

    Handles ``#`` comments and quoted scalars — enough for the alias file when
    PyYAML is absent.  Not a general YAML parser.
    """
    data: dict = {}
    current = None
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0] if " #" in raw else raw
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current is not None:
            data[current].append(_unquote(stripped[2:]))
            continue
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            data[key] = [_unquote(x) for x in val[1:-1].split(",") if x.strip()]
            current = None
        elif val:
            data[key] = _unquote(val)
            current = None
        else:
            data[key] = []
            current = key
    return data


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


__all__ = [
    "load",
    "load_aliases",
    "load_input_format_aliases",
    "load_ignored_fields",
    "load_transformer_typing",
    "load_diffusion_aliases",
    "load_diffusion_typing",
    "load_diffusion_text_encoders",
    "load_diffusion_conditioning",
]
