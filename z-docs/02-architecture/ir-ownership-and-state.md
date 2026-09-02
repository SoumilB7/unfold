# IR, ownership, and epistemic state

## ModelIR

`unfold-pkg/model_unfolder/ir.py` defines `AttentionSpec`, `FFNSpec`,
`LayerSpec`, and `ModelIR`. The IR supports heterogeneous layers and carries
domain extras needed by modalities, diffusion, evidence, and rendering.

The IR is an active conversion boundary. It contains both architectural values
and fields that describe how values were established. New work must prefer a
typed evidence fact plus a clear projection contract over adding another raw
flag to `extras`.

## Owner paths

Every claim belongs to an exact component and mechanism, for example:

```text
root
root.text
root.vision
root.audio
root.vae
root.denoiser
decoder.attention
decoder.ffn
```

The exact path may be deeper. Owners with the same leaf field are not
interchangeable. A vision `hidden_size` cannot discharge a text
`hidden_size` obligation.

## Evidence states

The current fact system recognizes states including:

- `code_proven`;
- `code_and_config`;
- `config_declared`;
- `class_default`;
- `derived`;
- `ambiguous`;
- `oracle_missing`;
- `unknown`;
- temporary asserted states tracked for removal.

The definitions are in `model_unfolder/evidence/context.py` and
`model_unfolder/evidence/facts.py`.

## Strength law

A derived fact cannot be stronger than its weakest premise. A negative
code-proven fact requires complete inspection. A missing source or reader error
cannot decorate a proven claim. `EvidenceFact` refuses truthiness so callers
cannot write `fact or default` and accidentally erase epistemic state.

## Unknown-safe geometry

Geometry that cannot be established is optional, not zero. Every consumer must
handle the unknown state explicitly. Parameter formulas that require unknown
geometry return an incomplete result rather than calculating with a placeholder.

## Component applicability

Ownership and applicability are facts upstream of the renderer. Recursive
parses retain their ambient namespace, so a modality inside
`root.text_encoder` cannot become the pipeline's top-level `root.vision`.
Candidate discovery is not ownership, and ownership is not reachability.
