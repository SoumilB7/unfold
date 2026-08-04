"""Blocking inventory of surviving pre-ProgramIndex semantic readers.

This register is debt, not architecture vocabulary.  It prevents the U3
completion amendment from becoming a loophole: every surviving raw-files reader
has one exact definition, exact production callers, one future owner unit and a
checkable deletion condition.  New/renamed/moved readers or callers fail the
gate.  The assigned U6-U11 unit deletes its row with the reader.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


_VALID_UNITS = frozenset({"U7", "U8", "U10", "U11"})

# Line-insensitive AST/content pins.  They freeze the registered reader bodies
# plus their same-module helper closure, and the exact callers of every legacy
# model-source parse authority.  A future owning unit updates/shrinks these only
# in the same commit that deletes or migrates the old authority.
LEGACY_READER_IMPLEMENTATION_FINGERPRINT = (
    "01da018088949ea2fc2e081a7e127dd916e4bb8f3e0f8019f3aad254b0424264")
LEGACY_PARSE_CALLER_FINGERPRINT = (
    "3c1efd6a9da25584b2a89a1b241e2a310f044dbdc5469f2f4ec40a5062622d55")


@dataclass(frozen=True)
class LegacySemanticReader:
    definition_path: str
    symbol: str
    migration_unit: str
    reason: str
    deletion_condition: str
    callers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.definition_path.startswith("model_unfolder/evidence/") \
                or not self.definition_path.endswith(".py"):
            raise ValueError("legacy reader definitions are exact evidence-module paths")
        if not self.symbol.endswith("_from_files"):
            raise ValueError("legacy reader symbols retain their exact debt spelling")
        if self.migration_unit not in _VALID_UNITS:
            raise ValueError("legacy readers are assigned to U7/U8/U10/U11")
        if not self.reason or not self.deletion_condition:
            raise ValueError("legacy reader debt needs reason + deletion condition")
        if tuple(sorted(set(self.callers))) != self.callers:
            raise ValueError("legacy reader callers are a canonical exact set")


@dataclass(frozen=True, order=True)
class SourceParseSite:
    path: str
    enclosing_symbol: str

    def __post_init__(self) -> None:
        if not self.path.startswith("model_unfolder/evidence/") \
                or not self.path.endswith(".py"):
            raise ValueError("source parse sites are exact evidence-module paths")
        if not self.enclosing_symbol:
            raise ValueError("source parse sites carry an enclosing symbol")


@dataclass(frozen=True)
class ParseAuthority:
    site: SourceParseSite
    category: str
    deletion_unit: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        categories = {
            "central_program_index", "address_bootstrap", "repository_audit",
            "test_guard", "legacy_model_source",
        }
        if self.category not in categories:
            raise ValueError(f"unknown parse-authority category {self.category!r}")
        if self.category == "legacy_model_source" \
                and (not self.deletion_unit or not self.reason):
            raise ValueError("legacy model-source parses need deletion unit + reason")
        if self.category != "legacy_model_source" and self.deletion_unit:
            raise ValueError("lawful parse sites do not carry a deletion unit")


def _row(symbol: str, unit: str, reason: str, *, module: str = "patterns.py",
         callers: tuple[str, ...] = ()) -> LegacySemanticReader:
    return LegacySemanticReader(
        f"model_unfolder/evidence/{module}",
        symbol,
        unit,
        reason,
        f"{unit} registers the exact owner-qualified fact and deletes {symbol}",
        callers,
    )


LEGACY_SEMANTIC_READERS = (
    _row("attention_score_scaling_from_files", "U10",
         "whole-file diffusion-attention score-scaling interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_scores_scaled",
         )),
    _row("decoder_layer_topology_from_files", "U7",
         "whole-file decoder cell-topology interpretation",
         callers=(
             "model_unfolder/adapters/transformer/parser.py:_code_layer_topology",
         )),
    _row("decoder_router_evidence_from_files", "U7",
         "whole-file router/mechanism interpretation",
         callers=(
             "model_unfolder/adapters/transformer/parser.py:_code_router",
         )),
    _row("layer_class_count_from_files", "U7",
         "whole-file layer-class counting used as topology evidence",
         callers=(
             "model_unfolder/evidence/validate.py:_looks_like_multi_variant_file",
         )),
    _row("attention_causality_from_files", "U8",
         "whole-file mask/causality interpretation",
         callers=(
             "model_unfolder/adapters/transformer/parser.py:_code_attention_causality",
         )),
    _row("decoder_moe_schedule_from_files", "U8",
         "whole-file per-layer MoE selector interpretation",
         callers=(
             "model_unfolder/adapters/transformer/parser.py:_code_moe_schedule",
         )),
    _row("decoder_rope_dim_from_files", "U8",
         "whole-file positional dimension interpretation",
         callers=(
             "model_unfolder/adapters/transformer/parser.py:_code_rope_dim",
         )),
    _row("denoiser_block_timestep_conditioning_from_files", "U10",
         "diffusion block conditioning interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_block_conditioning",
         )),
    _row("diffusion_attn_kind_from_files", "U10",
         "diffusion attention mechanism interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_attn_kind",
         )),
    _row("diffusion_axes_dims_rope_from_files", "U10",
         "diffusion positional axes interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_axes_dims_rope",
         )),
    _row("diffusion_cross_qk_norm_from_files", "U10",
         "diffusion cross-attention QK norm interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_cross_qk_norm",
         )),
    _row("diffusion_ffn_activation_from_files", "U10",
         "diffusion FFN activation interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_ffn_activation",
         )),
    _row("diffusion_ffn_kind_from_files", "U10",
         "diffusion FFN mechanism interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_ffn_kind",
         )),
    _row("diffusion_gate_via_norm_from_files", "U10",
         "diffusion modulation/gating interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_gate_via_norm",
         )),
    _row("diffusion_qk_norm_from_files", "U10",
         "diffusion self-attention QK norm interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_qk_norm",
         )),
    _row("diffusion_rope_from_files", "U10",
         "diffusion positional mechanism interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_has_rope",
         )),
    _row("diffusion_single_stream_fusion_from_files", "U10",
         "diffusion stream/fusion interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_single_fusion",
         )),
    _row("secondary_stacks_from_files", "U10",
         "diffusion repeated-stack topology interpretation", module="stacks.py",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_code_block_norm_placement",
             "model_unfolder/adapters/diffusor/parser.py:_code_norm_kind",
             "model_unfolder/adapters/diffusor/parser.py:_secondary_stack_specs",
         )),
    _row("unet_code_attention_placement_from_files", "U11",
         "UNet attention placement interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_parse_unet_model",
         )),
    _row("unet_mid_block_present_from_files", "U11",
         "UNet mid-stage topology interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_parse_unet_model",
         )),
    _row("unet_stage_attn_cell_from_files", "U11",
         "UNet stage attention-cell interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_parse_unet_model",
         )),
    _row("unet_stage_temporal_from_files", "U11",
         "UNet temporal-cell interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_parse_unet_model",
         )),
    _row("unet_transformer_ffn_activation_from_files", "U11",
         "UNet nested-transformer FFN interpretation",
         callers=(
             "model_unfolder/adapters/diffusor/parser.py:_parse_unet_model",
         )),
)


def _parse(path: str, symbol: str, category: str, unit: str = "",
           reason: str = "") -> ParseAuthority:
    return ParseAuthority(SourceParseSite(path, symbol), category, unit, reason)


PARSE_AUTHORITY_SITES = (
    _parse("model_unfolder/evidence/program_index.py", "_observe_source",
           "central_program_index"),
    _parse("model_unfolder/evidence/sources.py", "_architecture_from_config_class",
           "address_bootstrap"),
    _parse("model_unfolder/evidence/identity_guard.py", "scan_identity_source",
           "repository_audit"),
    _parse("model_unfolder/evidence/structural_debt.py", "_module_symbols",
           "repository_audit"),
    _parse("model_unfolder/evidence/structural_writes.py", "_scan_raw",
           "repository_audit"),
    _parse("model_unfolder/evidence/consumer_firewall.py",
           "scan_consumer_source", "repository_audit"),
    _parse("model_unfolder/evidence/legacy_reader_quarantine.py",
           "observed_evidence_parse_sites", "repository_audit"),
    _parse("model_unfolder/evidence/legacy_reader_quarantine.py",
           "_observed_callers", "repository_audit"),
    _parse("model_unfolder/evidence/legacy_reader_quarantine.py",
           "observed_reader_definitions", "repository_audit"),
    _parse("model_unfolder/evidence/legacy_reader_quarantine.py",
           "observed_reader_implementation_fingerprint", "repository_audit"),
    _parse("model_unfolder/evidence/forward_ops.py",
           "unclassified_call_tokens", "test_guard"),
    _parse("model_unfolder/evidence/ast_scanner.py", "scan_python_files",
           "legacy_model_source", "U14",
           "parallel class/config scan used by conformance and legacy adapters"),
    _parse("model_unfolder/evidence/audio.py", "_class_node",
           "legacy_model_source", "U9",
           "audio evidence reparses its component source"),
    _parse("model_unfolder/evidence/conformance.py", "_constructor_envs",
           "legacy_model_source", "U14",
           "conformance-local constructor interpreter"),
    _parse("model_unfolder/evidence/conformance.py", "_imported_model_files",
           "legacy_model_source", "U14",
           "conformance-local import closure"),
    _parse("model_unfolder/evidence/conformance.py",
           "_init_helper_block_classes", "legacy_model_source", "U14",
           "conformance-local helper/block scan"),
    _parse("model_unfolder/evidence/conformance.py", "_selected_init_refs",
           "legacy_model_source", "U14",
           "conformance-local constructor selection"),
    _parse("model_unfolder/evidence/forward_ops.py", "_parse_file",
           "legacy_model_source", "U14",
           "parallel forward-op parser shared by legacy facts/conformance"),
    _parse("model_unfolder/evidence/fusion.py", "_class_node",
           "legacy_model_source", "U9",
           "fusion evidence reparses its component source"),
    _parse("model_unfolder/evidence/patterns.py", "_find_decoder_layer",
           "legacy_model_source", "U8",
           "shared transformer semantic class selector"),
    _parse("model_unfolder/evidence/patterns.py", "_parse_defs",
           "legacy_model_source", "U8",
           "shared transformer/diffusion semantic parser"),
    _parse("model_unfolder/evidence/patterns.py",
           "attention_causality_from_files", "legacy_model_source", "U8",
           "whole-file causality interpreter"),
    _parse("model_unfolder/evidence/patterns.py",
           "attention_score_scaling_from_files", "legacy_model_source", "U10",
           "whole-file diffusion score-scaling interpreter"),
    _parse("model_unfolder/evidence/patterns.py",
           "decoder_router_evidence_from_files", "legacy_model_source", "U7",
           "whole-file router interpreter"),
    _parse("model_unfolder/evidence/patterns.py",
           "diffusion_axes_dims_rope_from_files", "legacy_model_source", "U10",
           "diffusion positional interpreter"),
    _parse("model_unfolder/evidence/patterns.py",
           "diffusion_gate_via_norm_from_files", "legacy_model_source", "U10",
           "diffusion modulation interpreter"),
    _parse("model_unfolder/evidence/patterns.py",
           "diffusion_qk_norm_from_files", "legacy_model_source", "U10",
           "diffusion normalization interpreter"),
    _parse("model_unfolder/evidence/patterns.py",
           "diffusion_single_stream_fusion_from_files", "legacy_model_source",
           "U10", "diffusion stream interpreter"),
    _parse("model_unfolder/evidence/patterns.py",
           "layer_class_count_from_files", "legacy_model_source", "U7",
           "whole-file layer/topology counter"),
    _parse("model_unfolder/evidence/position.py", "_call_line",
           "legacy_model_source", "U8",
           "position reader reparses source for a call span"),
    _parse("model_unfolder/evidence/position.py", "_class_forward",
           "legacy_model_source", "U8",
           "position reader reparses source for a class forward"),
    _parse("model_unfolder/evidence/projector.py", "_class_node",
           "legacy_model_source", "U9",
           "projector evidence reparses its component source"),
    _parse("model_unfolder/evidence/transitive.py", "_parse_file",
           "legacy_model_source", "U14",
           "parallel callable/transitive parser shared by conformance"),
    _parse("model_unfolder/evidence/vision.py", "_parsed_classes",
           "legacy_model_source", "U9",
           "vision evidence reparses its component source"),
)


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted((root / "model_unfolder").rglob("*.py")))


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, path: str, watched: frozenset[str]):
        self.path = path
        self.watched = watched
        self.stack: list[str] = []
        self.alias_scopes: list[dict[str, str]] = [{}]
        self.calls: dict[str, set[str]] = {name: set() for name in watched}

    def _scope(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.alias_scopes.append({})
        self.generic_visit(node)
        self.alias_scopes.pop()
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.alias_scopes.append({})
        self.generic_visit(node)
        self.alias_scopes.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name in self.watched:
                self.alias_scopes[-1][alias.asname or alias.name] = alias.name

    def _reader_reference(self, value) -> str | None:
        if isinstance(value, ast.Name):
            return self._bound_reader(value.id)
        if isinstance(value, ast.Attribute) and value.attr in self.watched:
            return value.attr
        return None

    def _remember_alias(self, target, value) -> None:
        if isinstance(target, ast.Name):
            reader = self._reader_reference(value)
            if reader is not None:
                self.alias_scopes[-1][target.id] = reader

    def visit_Assign(self, node):
        for target in node.targets:
            self._remember_alias(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._remember_alias(node.target, node.value)
        self.generic_visit(node)

    def _bound_reader(self, name: str) -> str | None:
        for scope in reversed(self.alias_scopes):
            if name in scope:
                return scope[name]
        return name if name in self.watched else None

    def _record_reference(self, reader: str) -> None:
        self.calls[reader].add(f"{self.path}:{self._scope()}")

    def visit_Call(self, node):
        reader = None
        if isinstance(node.func, ast.Name):
            reader = self._bound_reader(node.func.id)
        elif isinstance(node.func, ast.Attribute) \
                and node.func.attr in self.watched:
            # Attribute qualification is deliberately conservative: any
            # production call of the quarantined exact symbol must be reviewed,
            # whether the receiver is a module alias or another indirection.
            reader = node.func.attr
        if reader is not None:
            self._record_reference(reader)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            reader = self._bound_reader(node.id)
            if reader is not None:
                self._record_reference(reader)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load) and node.attr in self.watched:
            self._record_reference(node.attr)
        self.generic_visit(node)


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.stack: list[str] = []
        self.definitions: dict[str, set[str]] = {}

    def _record(self, node) -> None:
        qualname = ".".join((*self.stack, node.name))
        if node.name.endswith("_from_files"):
            self.definitions.setdefault(node.name, set()).add(
                f"{self.path}:{qualname}")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _record
    visit_AsyncFunctionDef = _record

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


class _ParseVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.stack: list[str] = []
        self.ast_alias_scopes: list[set[str]] = [{"ast", "_ast"}]
        self.parse_alias_scopes: list[set[str]] = [set()]
        self.sites: set[SourceParseSite] = set()

    def _scope(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.ast_alias_scopes.append(set())
        self.parse_alias_scopes.append(set())
        self.generic_visit(node)
        self.parse_alias_scopes.pop()
        self.ast_alias_scopes.pop()
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.ast_alias_scopes.append(set())
        self.parse_alias_scopes.append(set())
        self.generic_visit(node)
        self.parse_alias_scopes.pop()
        self.ast_alias_scopes.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _is_ast_alias(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.ast_alias_scopes))

    def _is_parse_alias(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.parse_alias_scopes))

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "ast":
                self.ast_alias_scopes[-1].add(alias.asname or alias.name)

    def visit_ImportFrom(self, node):
        if node.module == "ast":
            for alias in node.names:
                if alias.name == "parse":
                    self.parse_alias_scopes[-1].add(alias.asname or alias.name)

    def _remember_alias(self, target, value) -> None:
        if not isinstance(target, ast.Name):
            return
        if isinstance(value, ast.Name):
            if self._is_ast_alias(value.id):
                self.ast_alias_scopes[-1].add(target.id)
            elif self._is_parse_alias(value.id):
                self.parse_alias_scopes[-1].add(target.id)
        elif isinstance(value, ast.Attribute) \
                and value.attr == "parse" \
                and isinstance(value.value, ast.Name) \
                and self._is_ast_alias(value.value.id):
            self.parse_alias_scopes[-1].add(target.id)

    def visit_Assign(self, node):
        for target in node.targets:
            self._remember_alias(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._remember_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        direct_alias = isinstance(func, ast.Name) \
            and self._is_parse_alias(func.id)
        attribute_alias = isinstance(func, ast.Attribute) \
                and func.attr == "parse" \
                and isinstance(func.value, ast.Name) \
                and self._is_ast_alias(func.value.id)
        if direct_alias or attribute_alias:
            self.sites.add(SourceParseSite(self.path, self._scope()))
        self.generic_visit(node)


def observed_reader_definitions(root: Path) -> dict[str, tuple[str, ...]]:
    found: dict[str, set[str]] = {}
    for path in _python_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        visitor = _DefinitionVisitor(rel)
        visitor.visit(tree)
        for name, definitions in visitor.definitions.items():
            found.setdefault(name, set()).update(definitions)
    return {name: tuple(sorted(definitions))
            for name, definitions in found.items()}


def observed_reader_callers(root: Path) -> dict[str, tuple[str, ...]]:
    watched = frozenset(row.symbol for row in LEGACY_SEMANTIC_READERS)
    return _observed_callers(root, watched)


def _observed_callers(
        root: Path, watched: frozenset[str]) -> dict[str, tuple[str, ...]]:
    found: dict[str, set[str]] = {name: set() for name in watched}
    for path in _python_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        visitor = _CallVisitor(rel, watched)
        visitor.visit(tree)
        for name, callers in visitor.calls.items():
            found[name].update(callers)
    return {name: tuple(sorted(callers)) for name, callers in found.items()}


def observed_legacy_parse_callers(root: Path) -> dict[str, tuple[str, ...]]:
    watched = frozenset(
        row.site.enclosing_symbol for row in PARSE_AUTHORITY_SITES
        if row.category == "legacy_model_source")
    return _observed_callers(root, watched)


def _top_level_functions(tree) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _local_function_calls(node, known: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted({
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in known
    }))


def observed_reader_implementation_fingerprint(root: Path) -> str:
    """Normalized reader + same-module helper closure fingerprint.

    Source locations and formatting are excluded.  Semantic implementation
    changes, helper rewrites, moves, additions and deletions change the digest.
    """
    by_path: dict[str, list[str]] = {}
    for row in LEGACY_SEMANTIC_READERS:
        by_path.setdefault(row.definition_path, []).append(row.symbol)
    implementations: list[tuple[str, str, str]] = []
    for rel, roots in sorted(by_path.items()):
        path = root / rel
        if not path.is_file():
            implementations.extend(
                (rel, symbol, "<missing>") for symbol in sorted(roots))
            continue
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            implementations.extend(
                (rel, symbol, f"<unreadable:{type(exc).__name__}>")
                for symbol in sorted(roots))
            continue
        functions = _top_level_functions(tree)
        known = frozenset(functions)
        pending = list(sorted(roots))
        included: set[str] = set()
        while pending:
            symbol = pending.pop()
            if symbol in included or symbol not in functions:
                continue
            included.add(symbol)
            pending.extend(_local_function_calls(functions[symbol], known))
        for symbol in sorted(included):
            implementations.append((
                rel, symbol,
                ast.dump(functions[symbol], include_attributes=False),
            ))
    payload = json.dumps(
        implementations, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def observed_legacy_parse_caller_fingerprint(root: Path) -> str:
    payload = json.dumps(
        observed_legacy_parse_callers(root),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def observed_evidence_parse_sites(root: Path) -> tuple[SourceParseSite, ...]:
    found: set[SourceParseSite] = set()
    for path in sorted((root / "model_unfolder" / "evidence").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        visitor = _ParseVisitor(rel)
        visitor.visit(tree)
        found.update(visitor.sites)
    return tuple(sorted(found))


def legacy_reader_quarantine_problems(root: Path) -> tuple[str, ...]:
    definitions = observed_reader_definitions(root)
    callers = observed_reader_callers(root)
    rows = {row.symbol: row for row in LEGACY_SEMANTIC_READERS}
    problems: list[str] = []

    if len(rows) != len(LEGACY_SEMANTIC_READERS):
        problems.append("duplicate legacy semantic reader row")
    for name in sorted(set(definitions) - set(rows)):
        problems.append(f"unregistered legacy semantic reader: {name}")
    for name in sorted(set(rows) - set(definitions)):
        problems.append(f"dead legacy semantic reader row: {name}")
    for name, row in sorted(rows.items()):
        actual_defs = definitions.get(name, ())
        expected_defs = (f"{row.definition_path}:{row.symbol}",)
        if actual_defs != expected_defs:
            problems.append(
                f"{name} definition drift: expected {expected_defs!r}, "
                f"observed {actual_defs!r}")
        actual_callers = callers.get(name, ())
        if actual_callers != row.callers:
            problems.append(
                f"{name} caller drift: expected {row.callers!r}, "
                f"observed {actual_callers!r}")
    expected_parse_sites = tuple(sorted(
        authority.site for authority in PARSE_AUTHORITY_SITES))
    actual_parse_sites = observed_evidence_parse_sites(root)
    if actual_parse_sites != expected_parse_sites:
        expected_set = set(expected_parse_sites)
        actual_set = set(actual_parse_sites)
        for site in sorted(actual_set - expected_set):
            problems.append(f"unregistered evidence ast.parse site: {site}")
        for site in sorted(expected_set - actual_set):
            problems.append(f"dead evidence ast.parse row: {site}")
    reader_fingerprint = observed_reader_implementation_fingerprint(root)
    if reader_fingerprint != LEGACY_READER_IMPLEMENTATION_FINGERPRINT:
        problems.append(
            "legacy reader/helper implementation drift: expected "
            f"{LEGACY_READER_IMPLEMENTATION_FINGERPRINT}, "
            f"observed {reader_fingerprint}")
    parse_caller_fingerprint = observed_legacy_parse_caller_fingerprint(root)
    if parse_caller_fingerprint != LEGACY_PARSE_CALLER_FINGERPRINT:
        problems.append(
            "legacy parse-authority caller drift: expected "
            f"{LEGACY_PARSE_CALLER_FINGERPRINT}, "
            f"observed {parse_caller_fingerprint}")
    return tuple(problems)


__all__ = [
    "LEGACY_SEMANTIC_READERS",
    "LEGACY_PARSE_CALLER_FINGERPRINT",
    "LEGACY_READER_IMPLEMENTATION_FINGERPRINT",
    "LegacySemanticReader",
    "PARSE_AUTHORITY_SITES",
    "ParseAuthority",
    "SourceParseSite",
    "legacy_reader_quarantine_problems",
    "observed_evidence_parse_sites",
    "observed_legacy_parse_callers",
    "observed_legacy_parse_caller_fingerprint",
    "observed_reader_callers",
    "observed_reader_definitions",
    "observed_reader_implementation_fingerprint",
]
