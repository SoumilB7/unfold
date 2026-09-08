# Bounded independent diffusion test migration review

**ACCEPT the static migration within the reviewed scope. Actual focused execution
is still the executor's responsibility.** No pytest, model run or production edit
was performed for this review. Exact reviewed sources and pins are retained here.

All 25 original failed test functions are mapped to 25 existing replacements.
No other pre-existing test function in `test_diffusion.py` changed relative to
the a83 source. The additional typed-negative tests keep the original failures
under test rather than replacing all inputs with successful configurations.

Input checks (`input-equality.json`) use literal AST values from original
`a83d704:tests/test_diffusion.py` and `83140f1:test_support/__init__.py`, compared
with the current fixtures and JSON inputs:

- Both original rejected head dictionaries are exact; the synthetic first IF
  removes only `num_attention_heads`. Its omitted `attention_head_dim` is tested
  as the labelled class default 8, not an equivalent field rename or publisher
  assertion.
- Full Kandinsky JSON bytes equal the independently retained publisher config
  at revision `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3`; all four prepared input
  hashes match their provenance file. The published channels and scale/shift
  values have not been replaced by the old synthetic fixture.
- The moved dual-encoder literal is exactly the old test's `_text_encoder_configs`.
- Original incomplete SDXL is retained exactly. Its positive counterpart adds
  only the previously reviewed three constructor fields. SVD and the second IF
  literals remain exactly unchanged.

The new negatives at `tests/test_diffusion.py:66`, `:89`, and `:982` require a
captured construction result, `ConstructionFailed`, stage `construct`, and the
specific rejected field, NoneType construction error, or custom factory name.
An unrelated monitor failure cannot satisfy these assertions. Positive tests
require actual code-proven construction/shape facts and own occurrence cards;
they cannot pass solely on an opaque fallback or a generic unknown label.

The latest four head captures at `S8-head-input-actual-736fdab` confirm both exact
head failures and both positive constructions. Their saved qualified IR matches
the asserted Kandinsky image projection shapes `[24576,1280]` / `[768]`, and
synthetic IF projection `[4096,4096]` and labelled default 8. The previously
independently reviewed corrected SDXL, SVD, encoder bridge and second IF captures
support the migrated occurrence, own-shape and bounded source-route assertions.

The migration retains actual context argument routing without promoting it to
K/V semantics; temporal Conv3d/AlphaBlender population plus the proved temporal
result argument; the encoder bridge's own Linear shape and conditional call
ports; per-stage own convolution shapes; five proved ResNet fragments and
conditional arithmetic; and the qualified FFN operation opener. The mid-stage
test explicitly preserves distinct actual occurrences and reports open order.
It does not claim the former template's two sequential edges were re-proved.
That previously reviewed template retirement remains a named product limitation,
not a newly completed computation proof.

The view-census split changes no dispatcher or production routing. Ordinary
corpus dispatch must contain six new UNet runtime views and none of the five
declared differential-only layouts. An explicit legacy comparison then exercises
all five old layouts, checks real click coupling, and joins the two observed sets
for the unchanged total registry census. The existing fallback set is untouched;
no unexercised view is automatically exempted or stamped as drawn.

No concrete lost source-proven obligation, accidental input drift, monitor-error
false pass, or blanket production legacy loophole was found in this bounded
source review. This does not replace the pending actual assertion run, coverage
run, full suite, or preservation approval.
