"""Address the isolated instance builder from an already resolved source root.

Import addresses locate code. They never select an architectural mechanism.
This boundary neither downloads weights nor imports the model in the parent.
"""
from pathlib import Path

from .component_owner import require_resolved_component_root
from .document import PreparedDocument


def request_from_resolved_source(document, bundle, root, *, source_overrides=()):
    from physics.instance_inventory import BuildRequest

    if not isinstance(document, PreparedDocument):
        raise TypeError("instance construction requires the prepared document")
    root = require_resolved_component_root(root, caller="instance factory address")
    symbol = root.graph.root.symbol
    source = Path(symbol.source.canonical_path)
    addresses = set()
    for override in source_overrides:
        if source.resolve() == Path(override.path).resolve():
            if override.sha256 != symbol.source.content_fingerprint:
                raise ValueError("static reader and instance override select different source bytes")
            addresses.add(override.module)
    for import_root in bundle.import_roots.get(symbol.source.component_key, ()):
        try:
            relative = source.relative_to(import_root.path).with_suffix("")
        except ValueError:
            continue
        parts = relative.parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if all(part.isidentifier() for part in parts):
            addresses.add(".".join((import_root.package, *parts)))
    if len(addresses) != 1:
        raise ValueError("the resolved root lacks one exact supported import address")
    module = next(iter(addresses))
    for override in source_overrides:
        if override.module == module and override.sha256 != symbol.source.content_fingerprint:
            raise ValueError("static reader and instance override select different source bytes")
    return BuildRequest(
        config=document.checkpoint, framework="diffusers",
        factory_module=module, factory_qualname=symbol.qualified_name,
        factory_method="from_config", source_overrides=source_overrides)


def build_resolved_instance(document, bundle, root, *, source_overrides=()):
    from physics.instance_inventory import Failure, InventoryResult, inventory_in_subprocess

    try:
        request = request_from_resolved_source(document, bundle, root,
                                               source_overrides=source_overrides)
    except ValueError as exc:
        return InventoryResult("failed", failure=Failure(
            "ConfigurationFailed", "resolved_source_address", str(exc)))
    return inventory_in_subprocess(request)
