# Independent component-scope review

**Scope/existence correction passes the reviewed controls; RETURN the omitted primary input card.** This is an early review of four explicitly frozen working files, not the final integrated checkpoint or fresh full-page acceptance.

The replay uses saved actual SDXL Refiner fact values from `S8-generalization-293c2d9/sdxl-refiner/ordinary/`, projects them with the corrected component-scope branch, and renders the actual overview SVG. No model construction, source-reader campaign, pytest or production edits.

## Positive controls

- Denoiser-only handoffs now retain the denoiser and source input cards, without manufacturing a scheduler, VAE, encoder, noise/image boundary or sampling-loop expansion.
- Scheduler-only, VAE-only and encoder-only handoffs each retain the supplied component without inventing the others. The partial overview keeps the components visible separately; it does not assert unsupported connecting edges.
- The full-pipeline control retains its existing component cards and sampling-loop representation.
- The 11 displayed source-input labels are all actual parameters of the selected root `forward`; none is currently an invented helper-local formal.

These are projector/overview controls. They do not re-prove older outer-component mechanisms or replace the required fresh actual model pages.

## Concrete remaining gap

The actual root declares and reads `sample`, but the denoiser-only overview has no `sample` input card. It shows 11 conditioning/optional formals and omits the primary data input.

Earliest producer: the partial-component branch in `unet_projection.py` recursively collects only route dictionaries with `kind == "formal"`. Primary-region abstraction deliberately replaces the carried `sample` with `region_input`, so that scan loses the input's root declaration. The proof's `input_is_formal` boolean does not preserve its name for projection.

Minimum closure: expose the exact root-owned primary-input declaration from the proof and project it. Do not infer it from a model name or a conventional `sample` spelling. Future helper expansion must preserve callable ownership or substitute call-site inputs before collecting external root formals; recursively seeing a helper's local formal is not proof of an external input, even when its spelling happens to match a root parameter.

`results.json` records the exact declared root parameters, every projected input label, the missing-primary-input result and the five component controls. Standalone SVGs are saved for each case.

## Exact source snapshot

- `adapters/diffusor/parser.py`: `2da1f1dc41eaec94cf2defa7fc6a35df8dd7a645eaa77ea08bfdacd2dfcb1758`
- `adapters/diffusor/unet_projection.py`: `39027ed5586949ad407a5f57b645d694d5d64c3f9a4c670ed02b97f23516478d`
- `renderers/html/views_diffusion.py`: `5be3e118f2a1b0d058299de9c4ecd910d895406c3720363b0bdfe6a2ee8042b1`
- `expanded/loop.py`: `accf5e4d2756c70f35ec156c6bdbf749335c007a5bdcd3f298f82c0baa5c8075`

Paths above are under `model_unfolder/`. Full exact bytes are retained as `.py.txt` review snapshots. `replay_scope.py` asserts these hashes before imports and checks they remain unchanged afterward.

Original command, run from primary `unfold-pkg`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verification/receipts/S8-component-scope-independent/replay_scope.py > verification/receipts/S8-component-scope-independent/results.json
```

Exit 0 records all controls and the omission; it does not mean the omission was accepted. A later source edit requires a new replay with explicitly updated reviewed hashes, preserving this receipt.
