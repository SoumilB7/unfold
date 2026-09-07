"""Shared exact source fixture for stage execution and construction-summary controls."""
import textwrap
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution


ROOT = """
    from torch.nn import ModuleList
    from .factory import build

    class Root:
        def __init__(self, config):
            self.alpha = ModuleList([])
            self.omega = ModuleList([])
            self.bridge = build(config.bridge_type)
            for token in config.alpha_types:
                item = build(token)
                self.alpha.append(item)
            for token in config.omega_types:
                item = build(token)
                self.omega.append(item)

        def forward(self, value):
            saved = (value,)
            for first in self.alpha:
                value, branch = first(value)
                saved += branch
            value = self.bridge(value)
            for second in self.omega:
                side = saved[-1:]
                value = second(value, side)
            return value
"""


FACTORY = """
    class Alpha:
        def forward(self, value, side=None): return value, (value,)
    class Beta:
        def forward(self, value, side=None): return value, (value,)
    def build(token):
        if token == "one": return Alpha()
        return Beta()
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(tmp_path, root=ROOT, factory=FACTORY):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root_path = _write(package / "root.py", root)
    _write(package / "factory.py", factory)
    return SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (root_path,)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot("pkg", str(package)),)},
    )


def _read(bundle):
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    result = read_unet_stage_execution(construction, bundle, root)
    return result, construction, topology, root
