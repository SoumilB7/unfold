"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import hashlib
from pathlib import Path
from model_unfolder.evidence.program_index import ClassRecord, ProgramIndex, SourceFileNode, SourceId, SourceSpan, SymbolId
from model_unfolder.ir import ModelIR
from physics.instance_inventory import InstanceInventory, ModuleNode, PackageVersion, Provenance, RepetitionGroup, ResolvedClass, SourceFile


FP = "a" * 64


CONFIG_HASH = hashlib.sha256(b'{"width":4}').hexdigest()


CLASS = ResolvedClass("fixture.model", "Block")


def _inventory(count=2):
    source = SourceFile("fixture.model", "model.py", FP)
    provenance = Provenance(
        (PackageVersion("fixture", "1"),), (source,), CONFIG_HASH,
        ResolvedClass("fixture.model", "Model"), "fixture.Model",
        "fixture.Model(config)", {},
        {"python": "3.12", "platform": "test", "hash_seed": "0",
         "network": "test-denied", "hf_hub_offline": "1",
         "transformers_offline": "1", "diffusers_offline": "1"},
    )
    root = ModuleNode(
        "", ResolvedClass("fixture.model", "Model"), "fixture.model",
        (ResolvedClass("fixture.model", "Model"),), ("blocks",), (), {}, ())
    container = ModuleNode(
        "blocks", ResolvedClass("torch.nn.modules.container", "ModuleList"),
        "torch.nn.modules.container",
        (ResolvedClass("torch.nn.modules.container", "ModuleList"),),
        tuple(str(i) for i in range(count)), (), {}, ())
    blocks = tuple(ModuleNode(
        f"blocks.{i}", CLASS, CLASS.module, (CLASS,), (), (), {}, ())
        for i in range(count))
    return InstanceInventory(
        1, provenance, (root, container, *blocks),
        (RepetitionGroup("blocks", "b" * 64,
                         tuple(f"blocks.{i}" for i in range(count))),)
        if count > 1 else (),
        (),
    )


def _product_index():
    source = SourceId(str(Path("model.py").resolve()), FP, "root")
    symbol = SymbolId(source, "Block")
    return ProgramIndex(
        "fixture", source_nodes=(SourceFileNode(source),),
        classes=(ClassRecord(
            symbol, span=SourceSpan(source, 1, 0, 20, 0)),),
        fingerprint="f" * 64,
    )


def _product_ir(*, head=True):
    blocks = [{"id": "head", "kind": "output", "label": "Output"}] \
        if head else []
    return ModelIR(
        "fixture", "Fixture", 8, 4, None, None, [],
        extras={"render": {"model_blocks": blocks}},
    )
