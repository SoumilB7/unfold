"""The LOUD-miss nets — silence is never success.

Two failure modes used to be silent:

1. **A vanished op** — a bare call token in a ``forward()`` that the op
   vocabulary doesn't know contributes nothing to the presence set, so the
   conformance diff quietly thins.  The ratchet here pins the unknown-token
   set EMPTY over every source file the corpus resolves: a new HF helper
   spelling fails this test and becomes a conscious ``op_tokens.yaml`` row.
   (The sweep that seeded the vocabulary found exactly one real compute op
   hiding in the noise — ``tanh`` was unmapped — which is this net's point.)

2. **A missing oracle** — the conformance/evidence tests skip when modeling
   source is unavailable, so an environment without ``transformers`` /
   ``diffusers`` reported an all-green suite with the code↔diagram guarantees
   silently gone.  The floor test FAILS (never skips) when the oracles the
   corpus depends on cannot resolve.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_coverage as tc

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.forward_ops import unclassified_call_tokens


def _corpus_files() -> list[str]:
    files: set[str] = set()
    for _name, cfg in tc.CORPUS.items():
        try:
            files |= set(ParseContext.build(cfg, source="local").source_bundle.files)
        except Exception:
            continue                     # synthetic model types with no source
    return sorted(files)


def test_no_unclassified_call_tokens_over_corpus_sources():
    """Every bare call token in every corpus-resolved forward() is classified
    or consciously ignored — an unknown token can no longer thin a diff
    silently.  Fix a failure by adding the token to ``op_tokens.yaml`` (as a
    real op kind, or to ``ignore`` when it is provably plumbing) — never by
    weakening this net."""
    unknown = unclassified_call_tokens(_corpus_files())
    assert unknown == {}, (
        "unclassified forward() call tokens (add each to op_tokens.yaml as a "
        f"conscious decision): {sorted(unknown)}"
    )


def test_unclassified_token_net_fires_on_a_novel_op(tmp_path):
    """Negative control: a genuinely unknown compute call is reported, while
    forward-locals, self-field calls, in-file helpers and constructions are
    correctly excused."""
    path = tmp_path / "novel.py"
    path.write_text(
        "def local_helper(x):\n"
        "    return x\n"
        "class NovelBlock:\n"
        "    def __init__(self):\n"
        "        self.mlp = SomeMLP()\n"
        "    def forward(self, x):\n"
        "        fn = resolve(x)\n"      # `resolve` unknown; `fn` a local -> excused
        "        y = fn(x)\n"
        "        y = local_helper(y)\n"             # in-file helper -> excused
        "        y = self.mlp(y)\n"                 # field rail -> excused
        "        err = ValueError(y)\n"             # construction -> excused
        "        return frobnicate_scores(y)\n"     # UNKNOWN -> must be reported
    )
    unknown = unclassified_call_tokens((str(path),))
    assert set(unknown) == {"frobnicate_scores", "resolve"}
    assert unknown["frobnicate_scores"] == 1


def test_oracle_floor_transformers_and_diffusers_sources_resolve():
    """The code oracles the whole conformance surface depends on MUST resolve
    in this environment — a missing package fails loudly here instead of
    letting ~20 evidence tests skip into a false green."""
    llama = {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
             "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
             "vocab_size": 100}
    bundle = ParseContext.build(llama, source="local").source_bundle
    assert bundle.files, (
        "transformers modeling source did not resolve — the op/fact/wiring/"
        "nested conformance nets are silently skipping in this environment"
    )

    flux = {"_class_name": "FluxTransformer2DModel", "num_layers": 1,
            "num_single_layers": 1, "attention_head_dim": 64,
            "num_attention_heads": 4, "in_channels": 16,
            "joint_attention_dim": 128, "pooled_projection_dim": 64}
    bundle = ParseContext.build(flux, source="local").source_bundle
    assert bundle.files, (
        "diffusers modeling source did not resolve — diffusion conformance is "
        "silently skipping in this environment"
    )

    # The corpus itself must keep resolving a healthy oracle surface: at least
    # the non-synthetic families.  (Synthetic model types are the only allowed
    # no-source entries.)
    resolved = [name for name, cfg in tc.CORPUS.items()
                if (lambda b: bool(b.files))(
                    ParseContext.build(cfg, source="local").source_bundle)]
    assert {"dense_gated", "dit_mmdit", "dit_cross", "unet",
            "dit_hybrid_encoder", "dit_moe_encoder"} <= set(resolved), resolved
