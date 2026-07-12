"""Report identity-derived architectural decisions before they can hide.

This module deliberately starts as a reporting net.  It has two independent
axes:

* a small AST guard catches explicit identity predicates and the known helper/
  profile mechanisms that carry them into structure; and
* a differential guard pre-resolves source, removes semantic identity from the
  config, then parses through the *same adapter and SourceBundle*.  Any remaining
  structural difference therefore came from identity being used as a fact, not
  from losing the address needed to find source.

The static net is intentionally conservative.  It reports candidates for human
triage; Unit 9 makes it blocking only after the pinned debt set reaches zero.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .identity_roles import IDENTITY_ROLE_DECORATORS


IDENTITY_CONFIG_KEYS = frozenset({
    "model_type", "architectures", "_class_name", "_name_or_path",
    "name_or_path", "model_id", "repo_id", "family", "family_hint",
    "vision_family", "audio_family", "profile",
})

_IDENTITY_HELPERS = frozenset({
    "model_family_hint", "vision_family_hint", "audio_family_hint",
    "_class_default", "_guess_model_type_from_id",
})

_IDENTITY_NAMES = frozenset({
    "model_type", "_class_name", "model_id", "repo_id", "vision_family",
    "audio_family", "profile", "profile_title", "root_arch",
})

_CLASS_IDENTITY_NAMES = frozenset({
    "class_name", "block_class", "owner_class", "projector_class", "cls", "name",
})

# Domain/family words make a class-name substring architectural. Ordinary
# low-level role vocabulary (RMSNorm -> norm, Linear -> linear) remains allowed.
_CLASS_DOMAIN_MARKERS = frozenset({
    "vision", "audio", "video", "diffusion", "transformer", "unet", "dit",
    "projector", "resampler", "merger", "multimodal", "single-stream",
    "single_stream", "singlestream", "3d", "t5", "clip",
})

# Two DECLARED, lawful categories of class-name vocabulary (Unit 9 boundary).
# Both are still surfaced — as ``declared_class_vocabulary`` findings pinned by
# their own test — so adding a table or a use site remains a conscious act; they
# are simply no longer identity DEBT:
#
# * CODE-SHAPE markers classify a class that was ALREADY RESOLVED from the
#   model's own code evidence (the conformance registry reads which classes the
#   model's ``__init__`` constructs; the marker only names the ROLE of that
#   resolved class — vision/audio domain, drill↔mixer pairing, diffusers
#   processor refs, the single-stream fallback behind the field-name primary).
#   Same category as ``forward_ops`` type-roles (RMSNorm -> norm), which the
#   guard has always allowed.
_CODE_SHAPE_MARKER_TABLES = frozenset({
    "single_stream_class_markers", "component_class_markers",
    "drill_class_markers", "processor_markers",
})
# * DECLARED-COMPONENT markers read a diffusers config's OWN ``_class_name`` —
#   a constructor spec's declaration of which component/algorithm to build
#   (``FlowMatch…Scheduler`` declares the integrator exactly like
#   ``architectures[0]`` declares the modeling file). Reading the declaration
#   is reading the config, not looking up per-model facts.
_DECLARED_COMPONENT_TABLES = frozenset({
    "dit_class_markers", "scheduler_flow_matching_markers",
})
# * DECLARED-ROLE suffixes interpret the config's own ``architectures[]`` task
#   declaration (causal-LM / conditional-generation wrapper).  This is NOT
#   code-derived structure and is intentionally pinned as an interim config
#   declaration so it cannot disappear behind the zero-debt headline.
_DECLARED_ROLE_TABLES = frozenset({
    "causal_lm_suffixes", "wrapper_generation_suffixes",
})
_DECLARED_VOCABULARY_TABLES = (
    _CODE_SHAPE_MARKER_TABLES | _DECLARED_COMPONENT_TABLES | _DECLARED_ROLE_TABLES
)

_CLASS_MARKER_TABLES = _DECLARED_VOCABULARY_TABLES

# * DISPLAY vocabulary maps a class name to a human LABEL only (I-1 allows
#   displaying identity).  Shape-wise these look exactly like the forbidden
#   class->structure tables, so the shape nets below classify them via this
#   pinned set (the table key, or the file stem for a whole-file map) and
#   report them as ``display_vocabulary_table`` — pinned exactly by their own
#   test, so a NEW display table (or a structural table hiding behind a
#   display name) is a conscious, reviewed act, never silent.
_DISPLAY_VOCABULARY_TABLES = frozenset({"scheduler_display", "text_encoders"})

_FILE_ROOT = "<file-root>"

# Family-keyed fact tables (``norm_kind: {pixtral: RMSNorm}``): keyed by
# model_type, so a populated one is always identity DEBT.  The class-keyed
# marker/vocabulary tables are NO LONGER listed here — they are class-keyed by
# SHAPE and routed through the lawful-resource manifest below, so renaming or
# moving one cannot slip it past a fixed name list.
_ARCHITECTURAL_FACT_TABLES = frozenset({
    "norm_kind", "norm_placement", "parallel_residual", "no_rope",
    "axes_dims_rope", "qk_norm", "ffn_activation_fn", "single_stream_fusion",
    "rope_3d", "gate_via_norm", "cross_attn_norm", "self_attn_kind", "ffn_kind",
    "fusion_kind", "projector_ops", "vision_family", "audio_family",
})


# ------------------------------------------------------------------------- #
# §16.2 LAWFUL-RESOURCE MANIFEST
# ------------------------------------------------------------------------- #
# A class-keyed table is exempt from identity DEBT only if a manifest entry
# matches its ``(path, table)`` AND its content fingerprint.  No exemption rests
# on a filename or a table name alone: rename it, move it to another file, or add
# a single entry to it and the ``(path, table)`` key or the content fingerprint
# stops matching — so it falls back to DEBT until a human re-reviews it and
# re-pins the fingerprint here.  This replaces the previous three failures the
# audit found: (1) display maps pinned by NAME did not detect added entries;
# (2) a blanket ``conformance/`` directory exemption let any table hide there;
# (3) declared-vocabulary tables were exempt by a bare name set.
@dataclass(frozen=True)
class LawfulTable:
    path: str            # everchanging file, relative to the package parent
    table: str           # table name, or ``_FILE_ROOT`` for a whole-file map
    category: str        # code_shape | declared_component | declared_role | display
    consumers: frozenset # exact production readers (documented + enforceable)
    fingerprint: str     # sha256[:16] of the reviewed content (``_table_fingerprint``)


# Reader groups measured from the tree (grep of each table's loader) — the
# lawful consumers of each category.  A name-literal access from OUTSIDE the
# permitted set is reported (``unpermitted_vocabulary_consumer``).
_CONFORMANCE_READERS = frozenset({
    "model_unfolder/evidence/forward_ops.py", "model_unfolder/evidence/patterns.py",
    "model_unfolder/evidence/conformance.py", "model_unfolder/evidence/transitive.py",
    "model_unfolder/evidence/registry.py", "model_unfolder/everchanging/__init__.py",
})
_DIFFUSOR_READERS = frozenset({
    "model_unfolder/adapters/diffusor/parser.py", "model_unfolder/evidence/sources.py",
    "model_unfolder/everchanging/__init__.py",
})
_DISPLAY_READERS = frozenset({
    "model_unfolder/everchanging/__init__.py", "model_unfolder/lint.py",
    "model_unfolder/renderers/html/block_views/unet.py",
    "model_unfolder/adapters/diffusor/parser.py", "model_unfolder/adapters/diffusor/unet.py",
    "model_unfolder/adapters/diffusor/blocks.py",
})
_DECODERNESS_READERS = frozenset({
    "model_unfolder/evidence/decoderness.py", "model_unfolder/everchanging/__init__.py",
})

_LAWFUL_TABLES: tuple[LawfulTable, ...] = (
    # code-shape role vocabulary (RMSNorm -> norm class): classifies a class
    # ALREADY RESOLVED from construction — the category the guard always allowed,
    # previously blanket-exempt by the ``conformance/`` path, now each pinned.
    LawfulTable("model_unfolder/everchanging/conformance/conformance_map.yaml", "single_stream_class_markers", "code_shape", _CONFORMANCE_READERS, "aae2029c89d83f48"),
    LawfulTable("model_unfolder/everchanging/conformance/fact_markers.yaml", "linear_attn", "code_shape", _CONFORMANCE_READERS, "504bcdba298a4637"),
    LawfulTable("model_unfolder/everchanging/conformance/fact_markers.yaml", "conv_ffn", "code_shape", _CONFORMANCE_READERS, "52a5d67633cb2601"),
    LawfulTable("model_unfolder/everchanging/conformance/transitive.yaml", "causal_mask_call_tokens", "code_shape", _CONFORMANCE_READERS, "75996c6e29c8b2cc"),
    LawfulTable("model_unfolder/everchanging/conformance/transitive.yaml", "component_class_markers", "code_shape", _CONFORMANCE_READERS, "d3852984fbf8b042"),
    LawfulTable("model_unfolder/everchanging/conformance/transitive.yaml", "drill_class_markers", "code_shape", _CONFORMANCE_READERS, "4740691d26de09a8"),
    LawfulTable("model_unfolder/everchanging/conformance/transitive.yaml", "processor_markers", "code_shape", _CONFORMANCE_READERS, "18b6e66fa3fb5897"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "route", "code_shape", _CONFORMANCE_READERS, "4417c0333fc88ed7"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "attention", "code_shape", _CONFORMANCE_READERS, "1607839224fe5606"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "norm", "code_shape", _CONFORMANCE_READERS, "3e8161b9b9f2d64a"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "activation", "code_shape", _CONFORMANCE_READERS, "fa4df7183fb9453e"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "embedding", "code_shape", _CONFORMANCE_READERS, "9a3f48ad8286ce8b"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "ffn", "code_shape", _CONFORMANCE_READERS, "f1ff638270101d2f"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "conv", "code_shape", _CONFORMANCE_READERS, "3fdc556e8889cc9e"),
    LawfulTable("model_unfolder/everchanging/conformance/type_roles.yaml", "linear", "code_shape", _CONFORMANCE_READERS, "9c4117ea1874dea2"),
    # declared-component: reads a diffusers config's OWN _class_name declaration.
    LawfulTable("model_unfolder/everchanging/diffusor/typing.yaml", "dit_class_markers", "declared_component", _DIFFUSOR_READERS, "047ffe4c3b4247f6"),
    LawfulTable("model_unfolder/everchanging/diffusor/typing.yaml", "scheduler_flow_matching_markers", "declared_component", _DIFFUSOR_READERS, "9b9b539456201c6e"),
    # declared-role: reads the config's OWN architectures[] task declaration.
    LawfulTable("model_unfolder/everchanging/transformer/decoderness.yaml", "causal_lm_suffixes", "declared_role", _DECODERNESS_READERS, "17ceb3cfa506b47c"),
    LawfulTable("model_unfolder/everchanging/transformer/decoderness.yaml", "wrapper_generation_suffixes", "declared_role", _DECODERNESS_READERS, "7e70a5bde5c5ab56"),
    # display: class name -> human LABEL only (lawful under I-1).  Fingerprint
    # pins the population, so adding an entry (or hiding a structural key under a
    # display name) changes the hash and fails until re-reviewed.
    LawfulTable("model_unfolder/everchanging/diffusor/text_encoders.yaml", _FILE_ROOT, "display", _DISPLAY_READERS, "15c8fec2ef2986d1"),
    LawfulTable("model_unfolder/everchanging/diffusor/typing.yaml", "scheduler_display", "display", _DISPLAY_READERS, "3e613000ec3b1d39"),
)
_LAWFUL_BY_KEY: dict[tuple[str, str], LawfulTable] = {
    (t.path, t.table): t for t in _LAWFUL_TABLES
}


def _canon_str(obj: Any) -> str:
    """Order-insensitive canonical form of a table's content for fingerprinting.

    A dict is sorted by key; a list is order-normalized (a marker list is a SET
    of spellings, so reordering it is not a content change).  Adding, removing,
    or editing any entry changes the string and therefore the fingerprint."""
    if isinstance(obj, dict):
        return "{" + ",".join(f"{k!r}:{_canon_str(v)}"
                              for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))) + "}"
    if isinstance(obj, list):
        return "[" + ",".join(sorted(_canon_str(x) for x in obj)) + "]"
    return repr(obj)


def _table_fingerprint(content: Any) -> str:
    return hashlib.sha256(_canon_str(content).encode("utf-8")).hexdigest()[:16]


def _classify_class_keyed_table(path: str, table: str, content: Any,
                                line: int) -> IdentityViolation:
    """Classify one class-keyed table against the lawful-resource manifest.

    * no manifest entry for ``(path, table)``            -> ``identity_table`` DEBT
    * manifest entry but the fingerprint has drifted      -> ``unreviewed_table_change`` DEBT
    * manifest entry, fingerprint matches, display category -> ``display_vocabulary_table``
    * manifest entry, fingerprint matches, declared/shape  -> ``declared_vocabulary_table``
    """
    entry = _LAWFUL_BY_KEY.get((path, table))
    display_name = Path(path).stem if table == _FILE_ROOT else table
    if entry is None:
        return IdentityViolation(
            path, line, "identity_table",
            f"class-keyed table {display_name!r} may select structure by name — "
            "derive from construction/forward evidence, or register it in the "
            "lawful-resource manifest with a category, consumers and fingerprint")
    actual = _table_fingerprint(content)
    if actual != entry.fingerprint:
        return IdentityViolation(
            path, line, "unreviewed_table_change",
            f"lawful {entry.category} table {display_name!r} content changed "
            f"(fingerprint {actual} != reviewed {entry.fingerprint}) — a new/edited "
            "entry must be re-reviewed and the manifest fingerprint re-pinned")
    kind = ("display_vocabulary_table" if entry.category == "display"
            else "declared_vocabulary_table")
    return IdentityViolation(
        path, line, kind,
        f"lawful {entry.category} vocabulary {display_name!r} (manifest-pinned)")


@dataclass(frozen=True)
class IdentityViolation:
    path: str
    line: int
    kind: str
    detail: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.line}:{self.kind}"


@dataclass(frozen=True)
class NameBlindResult:
    structural_equal: bool
    original: dict[str, Any]
    scrubbed: dict[str, Any]

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(_changed_paths(self.original, self.scrubbed))


def scan_identity_source(source: str, *, path: str = "<memory>") -> list[IdentityViolation]:
    """Return explicit identity-to-structure candidates in one Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [IdentityViolation(path, exc.lineno or 1, "syntax", str(exc))]

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    findings: dict[tuple[int, str, str], IdentityViolation] = {}

    def add(node: ast.AST, kind: str, detail: str) -> None:
        item = IdentityViolation(path, getattr(node, "lineno", 1), kind, detail)
        findings[(item.line, item.kind, item.detail)] = item

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _IDENTITY_HELPERS:
                parent = parents.get(node)
                # Definitions/imports are not calls.  Every runtime call to one
                # of these helpers is debt until proven address/display-only.
                add(parent or node, "identity_helper", f"runtime call to {name}()")
        if isinstance(node, ast.Compare) and _class_domain_predicate(node):
            add(node, "class_identity_branch",
                "class-name/domain substring controls an architectural branch")

        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)
                      ) and _class_domain_predicate(node):
            add(node, "class_identity_branch",
                "class-name/domain substring controls an architectural branch")

        # H0 no-growth net (§16.2 E4/E5/E7): a Python literal keyed by class
        # names is a class->structure table wherever it sits and whatever it is
        # named.  Detection is single-ENTRY (one classish key is already a
        # table) and single-CAPITAL aware (``Attention``/``Mlp`` count), and it
        # covers dict COMPREHENSIONS — whose keys are dynamic, so the class
        # names live in the comprehension's own literal data.  Structure must
        # come from resolved code; a display map belongs in the pinned
        # lawful-resource manifest, never an inline literal.
        if isinstance(node, ast.Dict):
            classish = [key.value for key in node.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and _class_like(key.value)]
            if classish:
                add(node, "class_keyed_literal",
                    f"dict literal keyed by class names ({', '.join(classish[:3])}, …): "
                    "structure must be derived from resolved-source evidence; a display "
                    "map belongs in the pinned lawful-resource manifest")
        if isinstance(node, ast.DictComp):
            classish = _class_name_constants(node)
            if classish:
                add(node, "class_keyed_literal",
                    f"dict comprehension over class-name data ({', '.join(classish[:3])}, …): "
                    "a comprehension cannot launder a class->structure table past the guard")

        if isinstance(node, ast.Constant) and node.value in _CLASS_MARKER_TABLES:
            category = (
                "code-shape: classifies a class resolved from init evidence"
                if node.value in _CODE_SHAPE_MARKER_TABLES else
                "declared-role: reads the config's own architectures task declaration"
                if node.value in _DECLARED_ROLE_TABLES else
                "declared-component: reads the config's own _class_name declaration"
            )
            add(node, "declared_class_vocabulary",
                f"runtime access to declared class vocabulary {node.value!r} ({category})")

        if isinstance(node, (ast.If, ast.IfExp)):
            test = node.test
            names = _loaded_names(test)
            # §16.2 E6: the exemption is a TYPED marker the author applies
            # (@identity_address / @identity_display), NOT a function name the
            # guard hard-codes.  A new function is not silently exempt, and a
            # structural branch hiding in an address-named function is now debt.
            if _enclosing_function_decorators(node, parents) & IDENTITY_ROLE_DECORATORS:
                continue
            if names & _IDENTITY_NAMES or _calls_named(test, _IDENTITY_HELPERS):
                add(test, "identity_branch", "identity-derived predicate controls a branch")
            # H4 (§16.6) — the identity->structure TAINT signature: a class-name
            # value tested against ANY string (not only a domain marker) whose
            # branch WRITES A STRUCTURAL SINK (a spec/opgraph ctor, or a dict keyed
            # by a structural term).  This is the fabrication the domain-marker
            # predicate misses; lawful code-shape returns a role string/bool, not a
            # structural sink, so it is NOT caught (production is clean — 0).
            if (isinstance(node, ast.If)
                    and _compares_class_identity_to_string(test)
                    and not _class_domain_predicate(test)
                    and _branch_writes_structural_sink(node)):
                add(test, "class_identity_branch",
                    "a class-name comparison decides a structural sink — resolve the "
                    "class from construction and derive from its forward, never "
                    "fabricate structure from the name (H4 taint)")

        if isinstance(node, (ast.Dict, ast.Subscript)):
            text = ast.get_source_segment(source, node) or ""
            if "profile_title" in text and any(token in text for token in ("qwen", "mistral", "pixtral")):
                add(node, "identity_profile", "family profile selects rendered metadata")

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "profile_title" for target in targets):
                add(node, "identity_profile", "family profile selects rendered metadata")

    return sorted(findings.values(), key=lambda item: (item.path, item.line, item.kind, item.detail))


_DECLARED_VOCABULARY_KINDS = frozenset({
    "declared_class_vocabulary", "declared_vocabulary_table",
})

# Kinds excluded from DEBT because their exact population is pinned by its own
# test (declared class vocabulary; class-keyed display maps).
_PINNED_VOCABULARY_KINDS = _DECLARED_VOCABULARY_KINDS | {"display_vocabulary_table"}


def _scan_all_findings(root: str | Path | None = None) -> list[IdentityViolation]:
    """One walk over production sources + everchanging YAML — debt AND declared."""
    package = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for relative in ("adapters", "renderers", "evidence"):
        files.extend((package / relative).rglob("*.py"))
    # Root-level production modules (parser / opgraph / submodel / encoder_panel
    # / params / …) author IR facts and canonical regions — the H0 census places
    # conflation point C-8 (ffn_region) here.  The no-growth gate's negative
    # proof is only as strong as this walk, so they are scanned too (G-7: a
    # "zero debt" claim requires complete inspection of everywhere debt could
    # live).  All root modules are clean today; no exemptions.
    files.extend(package.glob("*.py"))
    # Source addressing is explicitly allowed, so evidence/sources.py is not a
    # production sink scan target.  The forbidden guesser is reported directly.
    findings: list[IdentityViolation] = []
    for file in sorted(set(files)):
        if file == Path(__file__).resolve():
            continue
        source = file.read_text(encoding="utf-8")
        rel = str(file.relative_to(package.parent))
        for item in scan_identity_source(source, path=rel):
            if item.path.endswith("evidence/sources.py"):
                if ("_guess_model_type_from_id" not in item.detail
                        and item.kind not in _DECLARED_VOCABULARY_KINDS):
                    continue
            findings.append(item)
    for file in sorted((package / "everchanging").rglob("*.yaml")):
        rel = str(file.relative_to(package.parent))
        # Canonical-field -> spelling lists legitimately reuse fact-table names
        # (norm_kind etc.), so the NAMED-table check must not read alias files —
        # but the class-keyed SHAPE net still does: a marker table cannot hide
        # in an aliases file either.
        named = file.name != "aliases.yaml"
        findings.extend(scan_identity_yaml_source(
            file.read_text(encoding="utf-8"), path=rel, include_named=named))
    return findings


def scan_identity_debt(root: str | Path | None = None) -> list[IdentityViolation]:
    """Identity DEBT only: findings whose mechanism lets a model's NAME select
    its drawn architecture.  Declared class vocabulary (code-shape roles over
    init-resolved classes; _class_name component declarations) and pinned
    display maps are excluded here and pinned separately by
    :func:`scan_declared_class_vocabulary` / :func:`scan_display_vocabulary`."""
    return [item for item in _scan_all_findings(root)
            if item.kind not in _PINNED_VOCABULARY_KINDS]


def scan_declared_class_vocabulary(root: str | Path | None = None) -> list[IdentityViolation]:
    """Every declared-vocabulary table and use site — pinned by its own test so
    a NEW table or a NEW runtime access is a conscious, reviewed act."""
    return [item for item in _scan_all_findings(root)
            if item.kind in _DECLARED_VOCABULARY_KINDS]


def scan_display_vocabulary(root: str | Path | None = None) -> list[IdentityViolation]:
    """Every class-keyed DISPLAY map (class name -> human label) — lawful under
    I-1 (identity may be displayed), but shape-identical to a forbidden
    structural table, so its exact population is pinned by its own test."""
    return [item for item in _scan_all_findings(root)
            if item.kind == "display_vocabulary_table"]


# H4 item 1 — the fact-provenance rule.  A fact that CLAIMS code or derived
# evidence may not be decided by a name.  The DECLARED tiers are exempt: reading
# the checkpoint's own ``architectures[]`` / ``_class_name`` declaration is the
# lawful config-declared path (I-2, G-4), and those reads are pinned separately
# as declared vocabulary.
_CODE_EVIDENCE_STATUSES = frozenset({"code_proven", "code_and_config", "derived"})

# The names that make a provenance string an IDENTITY citation: the raw config
# identity keys, plus the declared class-vocabulary and display table names (a
# code-proven fact must not be decided by reading one of those tables either).
_PROVENANCE_IDENTITY_NAMES = (
    IDENTITY_CONFIG_KEYS | _DECLARED_VOCABULARY_TABLES | _DISPLAY_VOCABULARY_TABLES
)


def _identity_names_in_source(source: str) -> set[str]:
    """Identity field / table names cited as a provenance SEGMENT.

    Sources look like ``config:model_type`` / ``reader_name`` / ``file.py:12``.
    Split on the separators that sit between a scope and a field (``: . / space``)
    and match a whole segment EXACTLY — so a reader named
    ``attention_causality_from_files`` never trips on the generic word
    ``architecture``, while ``config:model_type`` does trip on ``model_type``."""
    import re
    segments = {seg for seg in re.split(r"[:./\s]+", source or "") if seg}
    return segments & _PROVENANCE_IDENTITY_NAMES


def scan_fact_provenance_identity(provenance: dict) -> list[IdentityViolation]:
    """Fact-provenance rule (I-1, H4 item 1): every structural fact that CLAIMS
    code or derived evidence must cite resolved source, never a name.

    Consumes a serialized ``ir.extras['fact_provenance']`` dict (fact key ->
    ``{value, status, source?}``).  A ``code_proven`` / ``code_and_config`` /
    ``derived`` fact whose ``source`` names an identity field or a declared
    class-vocabulary table is a violation — the config_declared / class_default
    tiers are exempt because reading the checkpoint's own declaration is lawful.
    The corpus is expected to satisfy this (a real hit is a genuine identity
    leak to REPORT, never a test to relax)."""
    findings: list[IdentityViolation] = []
    for key, row in (provenance or {}).items():
        if not isinstance(row, dict) or row.get("status") not in _CODE_EVIDENCE_STATUSES:
            continue
        hits = _identity_names_in_source(str(row.get("source") or ""))
        if hits:
            findings.append(IdentityViolation(
                "<ledger>", 0, "fact_provenance_identity",
                f"structural fact {key!r} ({row.get('status')}) cites identity "
                f"{sorted(hits)} as its deciding source — a code/derived fact must "
                "read resolved source; a name may only decide a config_declared fact"))
    return sorted(findings, key=lambda item: item.detail)


def scan_identity_yaml_source(source: str, *, path: str = "<memory>.yaml",
                              include_named: bool = True) -> list[IdentityViolation]:
    """Report populated family-keyed architectural fact tables in YAML.

    ``include_named=False`` runs only the class-keyed SHAPE net — for alias
    files whose keys legitimately reuse canonical fact-table names."""
    try:
        import yaml
        value = yaml.safe_load(source) or {}
    except Exception:
        # The project has a tiny fallback YAML dialect.  This line-oriented path
        # must represent every shape the nets below classify (lists, pair
        # lists, nested maps, root scalar maps), so the guard does not silently
        # weaken when PyYAML is absent.
        value = {}
        current = None
        for raw in source.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line:
                continue
            indented = line.startswith((" ", "\t"))
            stripped = line.strip()
            if not indented and not stripped.startswith("-") and ":" in stripped:
                key, _, scalar = stripped.partition(":")
                key, scalar = _bare(key), _bare(scalar)
                if scalar:                     # root ``key: scalar`` map row
                    value[key] = scalar
                    current = None
                else:                          # section header
                    current = key
                    value[current] = []
            elif current is not None and stripped.startswith("-"):
                if isinstance(value.get(current), list):
                    value[current].append(_bare(stripped[1:]))
            elif current is not None and indented and ":" in stripped:
                key, _, scalar = stripped.partition(":")
                if not isinstance(value.get(current), dict):
                    value[current] = {}
                value[current][_bare(key)] = _bare(scalar)

    findings: list[IdentityViolation] = []
    lines = source.splitlines()
    for key, table in (value.items() if isinstance(value, dict) and include_named else ()):
        # Family-keyed fact tables (``norm_kind: {pixtral: RMSNorm}``): keyed by
        # model_type, so a populated one is always identity DEBT.  Class-keyed
        # marker/vocabulary tables are handled by the SHAPE net + manifest below.
        if key not in _ARCHITECTURAL_FACT_TABLES or not table:
            continue
        line = next((i for i, text in enumerate(lines, 1) if text.startswith(f"{key}:")), 1)
        findings.append(IdentityViolation(
            path, line, "identity_table",
            f"populated architectural fact table {key!r} is keyed outside source evidence"))

    # §16.2 E3: there is NO blanket ``conformance/`` exemption.  Every class-keyed
    # table — in ANY file, conformance/ and aliases included — is classified by
    # SHAPE and checked against the lawful-resource manifest (its ``(path,
    # table)`` key AND its content fingerprint).  A code-role vocabulary is
    # exempt because it is REGISTERED with a category, consumers and a pinned
    # fingerprint, never because of the directory it happens to sit in.  Shapes,
    # in every spelling the everchanging loader supports:
    #   plain-list  key: [ClassA]      pair-list  key: ["ClassA=kind"]
    #   dict map    key: {ClassA: kind} file-root  ClassA: kind (whole file)
    # Detection is single-ENTRY and single-CAPITAL aware (§16.2 E4/E5).
    findings.extend(_class_keyed_shape_findings(value, lines, path))
    return findings


def _class_keyed_shape_findings(value, lines: list[str],
                                path: str) -> list[IdentityViolation]:
    """Every shape-detected class-keyed table, classified against the manifest.

    Yields ``declared_vocabulary_table`` / ``display_vocabulary_table`` for a
    registered table whose fingerprint matches, ``unreviewed_table_change`` when
    a registered table's content drifted, and ``identity_table`` DEBT for an
    unregistered one — so renaming, moving, or adding an entry to a lawful table
    all surface (§16.2 E2/E3/E4/E5)."""
    if not isinstance(value, dict):
        return []
    tables: list[tuple[str, Any, int]] = []   # (table name, content, line)

    # file-root map: ``ClassA: kind`` at top level (a whole-file table).  Single
    # entry is enough — one class->value row is already a class-keyed table.
    if any(isinstance(key, str) and _class_like(key)
           and isinstance(item, (str, int, float, bool))
           for key, item in value.items()):
        tables.append((_FILE_ROOT, value, 1))

    for key, table in value.items():
        if key in _ARCHITECTURAL_FACT_TABLES:
            continue
        line = next((i for i, text in enumerate(lines, 1)
                     if text.startswith(f"{key}:")), 1)
        if isinstance(table, dict):
            if any(isinstance(k, str) and _class_like(k) for k in table):
                tables.append((key, table, line))
        elif isinstance(table, list):
            # A class name may sit on EITHER side of a ``role=Class`` pair, and a
            # marker value may be a comma-joined substring list
            # (``vision=Vision,Visual``), so split on both and test every token.
            tokens = [_bare(part) for t in table if isinstance(t, str)
                      for part in t.replace("=", ",").split(",")]
            if any(_class_like(x) for x in tokens):
                tables.append((key, table, line))

    return [_classify_class_keyed_table(path, name, content, line)
            for name, content, line in tables]


def _bare(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _class_like(s: str) -> bool:
    """A PascalCase Python identifier used as a class name — single-CAPITAL
    aware (§16.2 E5), so ``Attention`` / ``Mlp`` (one capital) are not invisible.

    An initial capital plus at least one lowercase letter distinguishes a class
    name from: a snake_case field (``rms_norm`` — lowercase first), a human
    label (``CogVideoX DDIM`` — not an identifier), an all-caps constant
    (``ACT2FN`` — no lowercase), and a lowercase config value (``silu``)."""
    return bool(s) and s.isidentifier() and s[0].isupper() and any(c.islower() for c in s)


def _class_name_constants(node: ast.AST) -> list[str]:
    """Classish string constants anywhere in a subtree — for dict comprehensions,
    whose keys are dynamic so the class names live in the iterated literal data."""
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and _class_like(n.value)]


def _enclosing_function_decorators(node: ast.AST,
                                   parents: dict[ast.AST, ast.AST]) -> set[str]:
    """Decorator (simple) names on the nearest enclosing function — the typed
    address/display markers that lawfully exempt an identity branch (§16.2 E6)."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {_call_name(dec) for dec in current.decorator_list}
        current = parents.get(current)
    return set()


def scrub_semantic_identity(value: Any) -> Any:
    """Recursively remove names that may address code but cannot prove facts."""
    if isinstance(value, dict):
        return {
            key: scrub_semantic_identity(item)
            for key, item in value.items()
            if str(key) not in IDENTITY_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [scrub_semantic_identity(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_semantic_identity(item) for item in value)
    return value


def name_blind_diff(target: Any, *, context=None) -> NameBlindResult:
    """Compare structural IR with and without semantic identity.

    Adapter selection and source resolution happen from the original config.
    Both parses then use that exact adapter and pre-resolved context, preventing
    an address failure from masquerading as an architectural difference.
    """
    from ..adapters import find_adapter
    from ..parser import _coerce
    from .context import ParseContext

    cfg = _coerce(target)
    context = context or ParseContext.build(cfg, source="local")
    adapter = find_adapter(cfg)
    if adapter is None:
        raise ValueError("no adapter recognized the original config")

    original_ir = adapter.parse(cfg, context=context)
    scrubbed_ir = adapter.parse(scrub_semantic_identity(cfg), context=context)
    original = _normalized_structure(original_ir.to_dict())
    scrubbed = _normalized_structure(scrubbed_ir.to_dict())
    return NameBlindResult(original == scrubbed, original, scrubbed)


def _normalized_structure(value: dict[str, Any]) -> dict[str, Any]:
    """Drop presentation/address provenance while retaining architectural facts."""
    value = copy.deepcopy(value)
    value.pop("name", None)
    value.pop("architecture", None)
    # Warnings and access diagnostics describe evidence availability, not the
    # architecture.  Structural unknowns remain in the actual IR fields.
    value.pop("warnings", None)
    value.pop("notes", None)
    extras = value.get("extras")
    if isinstance(extras, dict):
        # config_audit / config_consumed / code_evidence are DIAGNOSTIC (which
        # fields were read / consumed / what the source scan found), not
        # architecture — dropped so a scrubbed-vs-original audit asymmetry
        # cannot trip the name-blind structural comparison.
        for key in ("config_audit", "config_consumed", "code_evidence"):
            extras.pop(key, None)
    _drop_display_class(value)
    return value


def _drop_display_class(value: Any) -> None:
    """``detail.class`` is the resolved class name shown on a card — display
    provenance (same category as the top-level ``name``/``architecture`` keys
    dropped above), never an architectural fact."""
    if isinstance(value, dict):
        detail = value.get("detail")
        if isinstance(detail, dict):
            detail.pop("class", None)
        for item in value.values():
            _drop_display_class(item)
    elif isinstance(value, list):
        for item in value:
            _drop_display_class(item)


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _calls_named(node: ast.AST, names: Iterable[str]) -> bool:
    names = set(names)
    return any(
        isinstance(item, ast.Call) and _call_name(item.func) in names
        for item in ast.walk(node)
    )


def _class_domain_predicate(node: ast.AST) -> bool:
    """True for a comparison that treats a class-name string as architecture.

    Literal comparisons (``"vision" in name``) are detected here. Dynamic
    vocabularies (``marker in cls``) are detected separately at their named
    marker-table access. Looking at an arbitrary enclosing call falsely labels
    ``_class_default(cls, "rope_3d")`` as a substring comparison.
    """
    if not any(
        _compare_uses_class_string(item)
        for item in ast.walk(node)
        if isinstance(item, ast.Compare)
    ):
        return False
    literals = {
        item.value.lower()
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    return any(
        _domain_marker_matches(marker, value)
        for value in literals
        for marker in _CLASS_DOMAIN_MARKERS
    )


def _compare_uses_class_string(node: ast.Compare) -> bool:
    return any(_is_class_string_expr(item) for item in (node.left, *node.comparators))


def _is_class_string_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _CLASS_IDENTITY_NAMES
    if isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr in {"lower", "casefold", "strip"}):
            return _is_class_string_expr(node.func.value)
        if isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
            return _is_class_string_expr(node.args[0])
    return False


# H4 (§16.6) — the identity->structure taint sink vocabulary.  A structural SINK
# is a dict keyed by one of these terms, or a spec/opgraph constructor; a branch
# that writes one of these BASED ON a class-name comparison is fabricating
# structure from a name.  Lawful code-shape returns a role string/bool instead.
_TAINT_STRUCT_KEYS = frozenset({
    "cell", "kind", "mixer", "norm_kind", "norm_placement", "gated", "activation",
    "mask", "projection_mode", "attention_kind", "ffn_kind", "fusion_kind",
    "position_kind", "self_attn_kind", "output_gate", "cross_kv_source",
    "cross_attention",
})
_TAINT_SPEC_CTORS = frozenset({"AttentionSpec", "FFNSpec", "LayerSpec", "Op", "Region"})


def _compares_class_identity_to_string(test: ast.AST) -> bool:
    """A comparison/membership test that pits a class-identity value against a
    string literal (``'X' in cls`` / ``cls == 'X'`` / ``cls.lower() == 'x'``)."""
    for cmp in ast.walk(test):
        if isinstance(cmp, ast.Compare) and _compare_uses_class_string(cmp):
            if any(isinstance(x, ast.Constant) and isinstance(x.value, str)
                   for x in ast.walk(cmp)):
                return True
    return False


def _branch_writes_structural_sink(if_node: ast.If) -> bool:
    """True when an ``if`` body constructs a structural sink — a dict literal keyed
    by a structural term, or a spec/opgraph constructor."""
    for n in ast.walk(if_node):
        if isinstance(n, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value in _TAINT_STRUCT_KEYS
                for k in n.keys):
            return True
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in _TAINT_SPEC_CTORS):
            return True
    return False


def _domain_marker_matches(marker: str, value: str) -> bool:
    """Match architectural words without short-token accidents (DiT/condition)."""
    if marker in {"dit", "t5", "3d"}:
        return marker == value or value.startswith(marker) or value.endswith(marker)
    return marker == value or marker in value


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _changed_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix]
        paths: list[str] = []
        for index, (a, b) in enumerate(zip(left, right)):
            paths.extend(_changed_paths(a, b, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def violation_snapshot(findings: Iterable[IdentityViolation]) -> str:
    """Stable JSON snapshot used while the guard is report-only."""
    return json.dumps([item.key for item in findings], indent=2)


__all__ = [
    "IDENTITY_CONFIG_KEYS", "IdentityViolation", "NameBlindResult",
    "name_blind_diff", "scan_declared_class_vocabulary", "scan_display_vocabulary",
    "scan_fact_provenance_identity", "scan_identity_debt", "scan_identity_source",
    "scan_identity_yaml_source", "scrub_semantic_identity", "violation_snapshot",
]
