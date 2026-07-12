"""The identity guard, BLOCKING — Unit 9 endpoint + §16.2 H0 repair.

Two properties, both blocking:

1. **Debt is zero and stays zero.** No identity-to-structure mechanism anywhere
   in production code (``scan_identity_debt() == []``).

2. **Every lawful class-keyed table is REGISTERED**, not name-exempt. A table is
   exempt from debt only if a lawful-resource-manifest entry matches its
   ``(path, table)`` AND its content fingerprint AND records its permitted
   consumers. No exemption rests on a filename or a table name alone — the three
   holes the independent audit found are each closed:

   * display maps were pinned by NAME and did not detect added entries
     -> the manifest fingerprints content, so growth fails until re-reviewed;
   * a blanket ``conformance/`` directory exemption let any table hide there
     -> removed; conformance tables are lawful because they are REGISTERED;
   * declared vocabulary was exempt by a bare name set
     -> replaced by the manifest (path + fingerprint + consumers).

The lawful exemption for an identity BRANCH is likewise a typed marker the author
applies (``@identity_address`` / ``@identity_display``), never a function name
the guard hard-codes.

Detection is single-ENTRY (one class-keyed row is already a table) and
single-CAPITAL aware (``Attention``/``Mlp``), and covers Python dict literals,
dict COMPREHENSIONS, and every YAML spelling the everchanging loader supports —
so renaming, relocating, shrinking, or growing a table cannot evade the net.

The tests below ARE the §16.2 H0 exit criteria — each poison control maps to one
requirement (manifest fingerprint, no blanket exemption, single-entry/-capital,
dict-comprehension, typed-wrapper branch exemption, the seven relocation poisons).
"""
from __future__ import annotations

import pathlib
from collections import Counter

import pytest

from model_unfolder.evidence.identity_guard import (
    _FILE_ROOT,
    _LAWFUL_BY_KEY,
    _LAWFUL_TABLES,
    _classify_class_keyed_table,
    _table_fingerprint,
    name_blind_diff,
    scan_declared_class_vocabulary,
    scan_display_vocabulary,
    scan_identity_debt,
    scan_identity_source,
    scan_identity_yaml_source,
)
from model_unfolder.evidence import identity_guard as _guard
from model_unfolder.evidence.identity_roles import IDENTITY_ROLE_DECORATORS

# unfold-pkg root (parent of the model_unfolder package) — manifest paths are
# relative to it, matching the production scan's ``file.relative_to(pkg.parent)``.
ROOT = pathlib.Path(_guard.__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# E12 — the headline: zero debt, blocking
# --------------------------------------------------------------------------- #

def test_identity_debt_is_zero_and_blocking():
    """No identity-derived architectural decision anywhere in production code.
    ANY new identity branch / helper / unregistered class-keyed table — or a
    drifted lawful table — fails this test."""
    assert scan_identity_debt() == []


# --------------------------------------------------------------------------- #
# E1 — the lawful-resource manifest is fully qualified (no name-only exemption)
# --------------------------------------------------------------------------- #

def test_manifest_entries_are_fully_qualified():
    """Every lawful table carries path, table, one of the four categories,
    NON-EMPTY permitted consumers, and a 16-hex content fingerprint — an
    exemption on a bare name is structurally impossible."""
    categories = {"code_shape", "declared_component", "declared_role", "display"}
    hexdigits = set("0123456789abcdef")
    for t in _LAWFUL_TABLES:
        assert t.path.startswith("model_unfolder/") and t.path.endswith(".yaml"), t
        assert t.table, t
        assert t.category in categories, t
        assert t.consumers, f"{t.table}: no permitted consumers recorded"
        assert all(c.startswith("model_unfolder/") for c in t.consumers), t
        assert len(t.fingerprint) == 16 and set(t.fingerprint) <= hexdigits, t
    keys = [(t.path, t.table) for t in _LAWFUL_TABLES]
    assert len(keys) == len(set(keys)), "duplicate (path, table) manifest keys"


# --------------------------------------------------------------------------- #
# E2 — content fingerprints are current, and match the tree exactly
# --------------------------------------------------------------------------- #

def test_lawful_fingerprints_are_current_and_match_tree():
    """Each manifest fingerprint equals the fingerprint recomputed from disk.
    Add, remove, or edit an entry in any lawful table and this fails until the
    change is reviewed and the fingerprint re-pinned (the display-growth hole)."""
    import yaml

    for t in _LAWFUL_TABLES:
        data = yaml.safe_load((ROOT / t.path).read_text(encoding="utf-8")) or {}
        content = data if t.table == _FILE_ROOT else data.get(t.table)
        assert content is not None, f"{t.table} not found in {t.path}"
        assert _table_fingerprint(content) == t.fingerprint, (
            f"{t.table} @ {t.path}: content drifted from the reviewed fingerprint "
            f"{t.fingerprint}; re-review and re-pin")


def test_every_lawful_table_is_visible_never_silently_skipped():
    """Each manifest entry surfaces on the real tree as a declared/display
    vocabulary finding — the guard never silently skips a lawful table (the
    ``never invisible`` property, including the ``role=Class`` marker tables)."""
    seen = {d.detail for d in scan_declared_class_vocabulary()}
    seen |= {d.detail for d in scan_display_vocabulary()}
    for t in _LAWFUL_TABLES:
        name = pathlib.Path(t.path).stem if t.table == _FILE_ROOT else t.table
        assert any(name in detail for detail in seen), f"{name} is invisible"


# --------------------------------------------------------------------------- #
# The Python name-literal access net (orthogonal, unchanged) — pinned exactly.
# --------------------------------------------------------------------------- #

EXPECTED_PYTHON_ACCESS = {
    ("model_unfolder/adapters/diffusor/parser.py",
     "runtime access to declared class vocabulary 'dit_class_markers' "
     "(declared-component: reads the config's own _class_name declaration)"): 1,
    ("model_unfolder/adapters/diffusor/parser.py",
     "runtime access to declared class vocabulary 'scheduler_flow_matching_markers' "
     "(declared-component: reads the config's own _class_name declaration)"): 1,
    ("model_unfolder/evidence/conformance.py",
     "runtime access to declared class vocabulary 'component_class_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 2,
    ("model_unfolder/evidence/conformance.py",
     "runtime access to declared class vocabulary 'drill_class_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 1,
    ("model_unfolder/evidence/conformance.py",
     "runtime access to declared class vocabulary 'processor_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 1,
    ("model_unfolder/evidence/conformance.py",
     "runtime access to declared class vocabulary 'single_stream_class_markers' "
     "(code-shape: classifies a class resolved from init evidence)"): 1,
    ("model_unfolder/evidence/sources.py",
     "runtime access to declared class vocabulary 'dit_class_markers' "
     "(declared-component: reads the config's own _class_name declaration)"): 1,
    ("model_unfolder/evidence/decoderness.py",
     "runtime access to declared class vocabulary 'causal_lm_suffixes' "
     "(declared-role: reads the config's own architectures task declaration)"): 1,
    ("model_unfolder/evidence/decoderness.py",
     "runtime access to declared class vocabulary 'wrapper_generation_suffixes' "
     "(declared-role: reads the config's own architectures task declaration)"): 1,
}


def test_python_declared_vocabulary_access_is_pinned():
    actual = Counter((i.path, i.detail) for i in scan_declared_class_vocabulary()
                     if i.kind == "declared_class_vocabulary")
    assert dict(actual) == EXPECTED_PYTHON_ACCESS


# --------------------------------------------------------------------------- #
# E4 — single-ENTRY class map (one class-keyed row is already a table)
# --------------------------------------------------------------------------- #

def test_single_entry_class_map_is_caught():
    yaml_one = scan_identity_yaml_source("m:\n  - Gemma3Sandwich=sandwich\n")
    assert any(f.kind == "identity_table" for f in yaml_one)

    py_one = scan_identity_source("X = {'FluxTransformer2DModel': 'dit'}\n",
                                  path="model_unfolder/adapters/x.py")
    assert any(f.kind == "class_keyed_literal" for f in py_one)


# --------------------------------------------------------------------------- #
# E5 — single-CAPITAL class name (Attention / Mlp — one capital)
# --------------------------------------------------------------------------- #

def test_single_capital_class_name_is_caught():
    poison = scan_identity_source("X = {'Attention': 'self', 'Mlp': 'ffn'}\n",
                                  path="model_unfolder/evidence/x.py")
    assert any(f.kind == "class_keyed_literal" for f in poison)

    # snake_case fields, all-caps constants, and lowercase config values are NOT
    # class names and must stay clean.
    clean = scan_identity_source("X = {'pre_norm': 1, 'ACT2FN': 2, 'silu': 3}\n")
    assert not any(f.kind == "class_keyed_literal" for f in clean)


# --------------------------------------------------------------------------- #
# E6 — the branch exemption is a TYPED marker, not a function name
# --------------------------------------------------------------------------- #

def test_no_function_name_exemption_set_exists():
    """The old ``_ADDRESS_OR_DISPLAY_FUNCTIONS`` name set is gone."""
    assert not hasattr(_guard, "_ADDRESS_OR_DISPLAY_FUNCTIONS")
    assert IDENTITY_ROLE_DECORATORS == {"identity_address", "identity_display"}


def test_identity_branch_is_debt_without_a_typed_marker():
    src = ("def choose(cfg):\n"
           "    model_type = cfg['model_type']\n"
           "    if model_type == 'pixtral':\n"
           "        return {'kind': 'rms'}\n")
    assert any(f.kind == "identity_branch" for f in scan_identity_source(src))


def test_identity_branch_exempt_only_via_typed_decorator():
    src = ("@identity_address\n"
           "def choose(cfg):\n"
           "    model_type = cfg['model_type']\n"
           "    if model_type == 'pixtral':\n"
           "        return locate_source(model_type)\n")
    assert not any(f.kind == "identity_branch" for f in scan_identity_source(src))


# --------------------------------------------------------------------------- #
# E7 — dict COMPREHENSION keyed by class-name data
# --------------------------------------------------------------------------- #

def test_dict_comprehension_is_caught():
    src = ("X = {n: k for n, k in "
           "[('FluxTransformer2DModel','dit'), ('VQModel','vq')]}\n")
    assert any(f.kind == "class_keyed_literal" for f in scan_identity_source(src))


# --------------------------------------------------------------------------- #
# E8 — a RENAMED / RELOCATED lawful table is not lawful
# --------------------------------------------------------------------------- #

def test_renamed_table_is_caught():
    """Lawful content under a NEW table name is debt — the (path, table) key no
    longer matches the manifest, so the name change is a conscious act."""
    y = scan_identity_yaml_source(
        "brand_new_role:\n  - RMSNorm\n  - LayerNorm\n",
        path="model_unfolder/everchanging/conformance/type_roles.yaml")
    assert any(f.kind == "identity_table" for f in y)


# --------------------------------------------------------------------------- #
# E9 — a helper that maps a class name to a structural kind
# --------------------------------------------------------------------------- #

def test_helper_returned_enum_is_caught():
    branch = scan_identity_source(
        "def kind(block_class):\n"
        "    if 'vision' in block_class.lower():\n"
        "        return 'vision_encoder'\n"
        "    return 'text'\n",
        path="model_unfolder/evidence/x.py")
    assert any(f.kind == "class_identity_branch" for f in branch)

    call = scan_identity_source("k = _class_default(cls, 'rope_3d')\n")
    assert any(f.kind == "identity_helper" for f in call)


# --------------------------------------------------------------------------- #
# E10 — structural data hidden UNDER a lawful display table (fingerprint drift)
# --------------------------------------------------------------------------- #

def test_structural_data_under_display_table_is_caught():
    entry = _LAWFUL_BY_KEY[
        ("model_unfolder/everchanging/diffusor/typing.yaml", "scheduler_display")]
    poisoned = ["CogVideoXDDIM=CogVideoX DDIM", "SneakyStructural=transformer2d"]
    finding = _classify_class_keyed_table(entry.path, "scheduler_display", poisoned, 1)
    assert finding.kind == "unreviewed_table_change"


# --------------------------------------------------------------------------- #
# E2 (growth) — adding one entry to a lawful display table fails until reviewed
# --------------------------------------------------------------------------- #

def test_display_map_growth_is_caught():
    import yaml

    entry = _LAWFUL_BY_KEY[
        ("model_unfolder/everchanging/diffusor/typing.yaml", "scheduler_display")]
    real = yaml.safe_load((ROOT / entry.path).read_text())["scheduler_display"]
    grown = list(real) + ["BrandNewScheduler=Brand New"]
    finding = _classify_class_keyed_table(entry.path, "scheduler_display", grown, 1)
    assert finding.kind == "unreviewed_table_change"


# --------------------------------------------------------------------------- #
# E11 — a mapping MOVED into conformance/ or an aliases file is caught
# --------------------------------------------------------------------------- #

def test_mapping_moved_into_conformance_is_caught():
    """The blanket conformance/ exemption is gone: a fresh class-keyed table
    dropped there is debt (only REGISTERED tables are lawful)."""
    y = scan_identity_yaml_source(
        "sneaky:\n  - UNetMidBlock2DCrossAttn=transformer2d\n",
        path="model_unfolder/everchanging/conformance/sneaky.yaml")
    assert any(f.kind == "identity_table" for f in y)


def test_mapping_moved_into_aliases_is_caught():
    """aliases.yaml keys legitimately reuse canonical fact-table names, so the
    NAMED check is off — but a class-keyed table cannot hide there: the SHAPE net
    still runs and it is not in the manifest."""
    y = scan_identity_yaml_source(
        "sneaky:\n  - AutoencoderKL=kl\n  - VQModel=vq\n",
        path="model_unfolder/everchanging/transformer/aliases.yaml",
        include_named=False)
    assert any(f.kind == "identity_table" for f in y)


# --------------------------------------------------------------------------- #
# End-to-end no-growth gate over a poisoned tree (kept from Unit 9)
# --------------------------------------------------------------------------- #

def test_no_growth_gate_fires_end_to_end_on_a_poisoned_tree(tmp_path):
    """An unsafe table added ANYWHERE the blocking zero-debt walk scans — any
    filename, any table name, YAML or Python — surfaces as identity DEBT."""
    package = tmp_path / "model_unfolder"
    for sub in ("adapters", "renderers", "evidence"):
        (package / sub).mkdir(parents=True)
    vocab_dir = package / "everchanging" / "newdomain"
    vocab_dir.mkdir(parents=True)
    (vocab_dir / "innocent.yaml").write_text(
        "helpful_hints:\n"
        "  - UNetMidBlock2DCrossAttn=transformer2d\n"
        "  - KDownBlock2D=resnet\n")
    (package / "adapters" / "helper.py").write_text(
        "KINDS = {'AutoencoderKL': 'kl', 'VQModel': 'vq'}\n")
    (package / "rootmod.py").write_text(
        "CELLS = {'UNetMidBlock2DCrossAttn': 'transformer2d',\n"
        "         'CrossAttnDownBlock2D': 'resnet'}\n")

    findings = scan_identity_debt(root=package)
    kinds = {item.kind for item in findings}
    assert "identity_table" in kinds
    assert "class_keyed_literal" in kinds
    literal_paths = {item.path for item in findings
                     if item.kind == "class_keyed_literal"}
    assert any(path.endswith("adapters/helper.py") for path in literal_paths)
    assert any(path.endswith("rootmod.py") for path in literal_paths)


# --------------------------------------------------------------------------- #
# Shape net: lawful controls stay clean; family fact tables are debt
# --------------------------------------------------------------------------- #

def test_shape_net_lawful_controls_stay_clean():
    """Snake-case field vocab and config-value enums must NOT become debt."""
    lanes = scan_identity_yaml_source(
        "stack_lane_params:\n  - encoder_hidden_states=text\n  - hidden_states=latent\n")
    assert not any(f.kind == "identity_table" for f in lanes)

    enums = scan_identity_yaml_source(
        "norm_type_kind:\n  - rms_norm=rmsnorm\n  - ada_norm=layernorm\n")
    assert not any(f.kind == "identity_table" for f in enums)

    ordinary = scan_identity_source("ok = {'pre': 1, 'post': 2}\n")
    assert not any(f.kind == "class_keyed_literal" for f in ordinary)


def test_family_fact_table_is_debt():
    """A model_type-keyed architectural fact table is always identity debt."""
    table = scan_identity_yaml_source(
        "norm_kind:\n  pixtral: RMSNorm\n  siglip: LayerNorm\n")
    assert any(f.kind == "identity_table" for f in table)

    debt = scan_identity_yaml_source("ffn_activation_fn:\n  pixtral: silu\n")
    assert any(f.kind == "identity_table" for f in debt)


def test_static_guard_catches_class_name_domain_substring_inside_evidence():
    findings = scan_identity_source(
        "def choose(block_class):\n"
        "    if 'vision' in block_class.lower():\n"
        "        return {'kind': 'vision_encoder'}\n",
        path="model_unfolder/evidence/new_detector.py")
    assert any(item.kind == "class_identity_branch" for item in findings)


def test_a_lawful_table_at_its_real_path_is_declared_not_debt():
    """The positive control for the manifest: the REAL scheduler_display content
    at its REAL path classifies as display vocabulary, not debt."""
    entry = _LAWFUL_BY_KEY[
        ("model_unfolder/everchanging/diffusor/typing.yaml", "scheduler_display")]
    import yaml
    real = yaml.safe_load((ROOT / entry.path).read_text())["scheduler_display"]
    finding = _classify_class_keyed_table(entry.path, "scheduler_display", real, 1)
    assert finding.kind == "display_vocabulary_table"


# --------------------------------------------------------------------------- #
# The differential (name-blind) axis — unchanged, strong, blocking
# --------------------------------------------------------------------------- #

def test_name_blind_guard_preserves_vision_structure_with_pre_resolved_source():
    pixtral_style = {
        "architectures": ["LlavaForConditionalGeneration"], "model_type": "llava",
        "image_token_index": 10, "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "mistral", "hidden_size": 5120, "num_hidden_layers": 4,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 14336, "vocab_size": 131072,
            "rms_norm_eps": 1e-5, "head_dim": 128,
        },
        "vision_config": {
            "model_type": "pixtral", "hidden_size": 1024, "image_size": 1024,
            "patch_size": 16, "num_hidden_layers": 24,
            "num_attention_heads": 16, "intermediate_size": 4096,
        },
    }
    result = name_blind_diff(pixtral_style)
    assert result.structural_equal
    assert result.changed_paths == ()


def test_name_blind_guard_preserves_source_address_and_clean_decoder_structure():
    from transformers import AutoConfig

    result = name_blind_diff(AutoConfig.for_model("llama").to_dict())
    assert result.structural_equal
    assert result.changed_paths == ()


def test_name_blind_guard_over_blessed_corpus():
    """BLOCKING corpus net, STRICT: every blessed fixture parses structurally
    IDENTICAL with all semantic identity scrubbed."""
    from model_unfolder.sable import load_corpus

    corpus = load_corpus()
    assert corpus, "no blessed fixtures — the corpus lock is gone"
    for fname, fix in corpus:
        result = name_blind_diff(fix["config"])
        assert result.structural_equal, \
            f"{fname}: name-blind structural drift: {result.changed_paths[:6]}"
