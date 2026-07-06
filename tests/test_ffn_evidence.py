"""Name-blind controls for exact feed-forward projection evidence."""
from model_unfolder.evidence.ffn import ffn_structure_evidence


def _write(tmp_path, class_name: str, *, fused: bool = False):
    fields = (
        "self.gate_up_proj = Linear()\n        self.down_proj = Linear()"
        if fused else
        "self.gate_proj = Linear()\n        self.up_proj = Linear()\n        self.down_proj = Linear()"
    )
    body = (
        "gate, up = self.gate_up_proj(x).chunk(2, dim=-1)"
        if fused else
        "gate = self.gate_proj(x)\n        up = self.up_proj(x)"
    )
    path = tmp_path / f"{class_name.lower()}.py"
    path.write_text(f'''\
class Linear:
    def __call__(self, x): return x
class GELU:
    def __call__(self, x): return x
class {class_name}:
    def __init__(self):
        {fields}
        self.act = GELU()
    def forward(self, x):
        {body}
        return self.down_proj(self.act(gate) * up)
''')
    return path


def test_ffn_structure_is_name_blind(tmp_path):
    a = ffn_structure_evidence((_write(tmp_path, "CompletelyNovelCell"),), expected_gated=True)
    b = ffn_structure_evidence((_write(tmp_path, "AnotherUnrelatedName"),), expected_gated=True)
    assert (a.status, a.gated, a.projection_mode) == ("proven", True, "split")
    assert (b.status, b.gated, b.projection_mode) == ("proven", True, "split")


def test_ffn_structure_distinguishes_fused_storage(tmp_path):
    evidence = ffn_structure_evidence(
        (_write(tmp_path, "NovelFusedCell", fused=True),), expected_gated=True,
    )
    assert evidence.status == "proven"
    assert evidence.gated is True and evidence.projection_mode == "fused_gate_up"


def test_ffn_structure_is_rooted_at_the_configured_owner(tmp_path):
    path = tmp_path / "mixed_component.py"
    path.write_text('''\
class Linear:
    def __call__(self, x): return x
class GELU:
    def __call__(self, x): return x
class DenseLeaf:
    def __init__(self):
        self.fc1 = Linear(); self.fc2 = Linear(); self.act = GELU()
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))
class UnrelatedGatedLeaf:
    def __init__(self):
        self.gate_proj = Linear(); self.up_proj = Linear(); self.down_proj = Linear(); self.act = GELU()
    def forward(self, x): return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
class ConfiguredOwner:
    def __init__(self): self.ff = DenseLeaf()
    def forward(self, x): return self.ff(x)
''')
    broad = ffn_structure_evidence((path,))
    exact = ffn_structure_evidence((path,), architecture="ConfiguredOwner")
    assert broad.status == "ambiguous"          # two genuine but unrelated layouts
    assert (exact.status, exact.owner_class, exact.projection_mode) == (
        "proven", "DenseLeaf", "dense",
    )


def test_factory_classmethod_construction_resolves_through_wrapper(tmp_path):
    """A wrapper that builds its tower via ``Tower._from_config(config)`` must
    still reach the tower's FFN leaf (the SD3.5/SDXL ``*WithProjection`` class
    of failure) — factory construction types the field as the BASE class."""
    path = tmp_path / "factory_wrapper.py"
    path.write_text('''\
class Linear:
    def __call__(self, x): return x
class GELU:
    def __call__(self, x): return x
class NovelDenseLeaf:
    def __init__(self):
        self.fc1 = Linear(); self.fc2 = Linear(); self.act = GELU()
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))
class NovelTower:
    def __init__(self):
        self.mlp = NovelDenseLeaf()
    def forward(self, x): return self.mlp(x)
class NovelTowerWithProjection:
    def __init__(self, config):
        self.tower = NovelTower._from_config(config)
        self.projection = Linear()
    def forward(self, x): return self.projection(self.tower(x))
''')
    evidence = ffn_structure_evidence(
        (path,), architecture="NovelTowerWithProjection",
    )
    assert (evidence.status, evidence.gated, evidence.projection_mode) == (
        "proven", False, "dense",
    ), evidence
    assert evidence.owner_class == "NovelDenseLeaf"


def test_factory_to_unknown_class_stays_ambiguous_not_fabricated(tmp_path):
    """A factory call to a class ABSENT from the files must stay unresolved —
    the closure may not borrow an unrelated leaf elsewhere in the file."""
    path = tmp_path / "factory_unknown.py"
    path.write_text('''\
class Linear:
    def __call__(self, x): return x
class GELU:
    def __call__(self, x): return x
class UnreachableGatedLeaf:
    def __init__(self):
        self.gate_proj = Linear(); self.up_proj = Linear(); self.down_proj = Linear(); self.act = GELU()
    def forward(self, x): return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
class OpaqueWrapper:
    def __init__(self, config):
        self.tower = NotInThisFile._from_config(config)
    def forward(self, x): return self.tower(x)
''')
    evidence = ffn_structure_evidence((path,), architecture="OpaqueWrapper")
    assert evidence.status == "ambiguous"
    assert evidence.gated is None
