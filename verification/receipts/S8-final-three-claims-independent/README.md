# Independent final SDXL three-claim review

Reviewed ordinary artifact: /private/tmp/unfold-s8-final-f664467/sdxl/ordinary. Corrected reporter commit e2dddf2, SHA256 5c696e8296f58b834d33204248f1610813a6709e4e0e9e6617a1bdb9b4c8e3cf. The reporter was imported with its file-writing function disabled; frozen campaign reports and product outputs were not overwritten. Exact selected traces, actual SVG fragments, source excerpts, shape recomputation and artifact pins are retained here.

Verdict: the three selected facts are supported at their stated scope, and all three corrected artifact-linkage chains have zero gaps. They do not represent three complete connected mechanism drawings. The final sheet must distinguish the following claims.

## 1. First FFN — ACCEPT as a connected computation trace

Occurrence down_blocks.1.attentions.0.transformer_blocks.0.ff; canonical fact root.denoiser.ffn_mechanisms, claim_kind applied_function, proof selected_composite_ffn_return_route.

Archived FeedForward source (attention SHA0192998d..., lines1725–1742) constructs the transform/dropout/output sequence and passes the evolving value through its selected modules. Archived GEGLU source (activations SHAab1767e8..., lines103–123) applies the input projection, splits gate/up, applies GELU to the gate and multiplies; the outer sequence then applies the output projection. The inventory selects GEGLU, exact Linear primitives at net.0.proj and net.2, and exact Dropout with p=0 at net.1. No class spelling alone is used as computation evidence.

The correct visible overview stage is instance_down_blocks__1. Its occurrence's own fact-citing card has runtime_ffn detail matching the fact. Actual drill SVG has the five operation nodes and seven arrows for the gate/up projection, split, GELU/multiply and down projection; its second SVG separately shows the constructed net container.

The actual card displays input projection5120×640, output projection640×2560 and4,920,960 parameters. Independent inventory arithmetic is5120×640+5120+640×2560+640=4,920,960. Exact weight/bias numbers also appear on the projection modules' own cards. This supports the selected FFN computation and its shapes, not every backend's numerical implementation or universal execution reachability.

## 2. First ResNet cell — ACCEPT only the five positive connection edges

Occurrence down_blocks.0.resnets.0; canonical fact root.denoiser.cell_connections, claim_kind connection, proof exact_constructed_member_call_result_to_argument. Its explicit coverage is positive_only.

Archived ResnetBlock2D source (SHA b601e45d..., lines326–365), together with this exact instance's upsample=None, downsample=None and time_embedding_norm='default', supports:

- norm1 → nonlinearity → conv1;
- norm2 → nonlinearity → dropout → conv2.

The correct visible overview stage is instance_down_blocks__0. The canonical card cites this fact. Its first two actual SVGs contain these two fragments with two and three arrows respectively, totaling the five claimed edges. Both activation invocations link to the same constructed SiLU module, while connection_calls retain distinct call IDs. The remaining SVGs show separate arithmetic fragments and containment.

The card explicitly says optional-branch/helper connections remain under investigation. There is no proved/drawn bridge from conv1 through conditioning to norm2 in this selected connection fact, and the return operands retain unknown lineage. Do not describe this trace as a fully connected residual block or a completed conditioning/residual path.

Its2,255,040 parameters are independently recovered as two921,920-parameter convolutions, a409,920-parameter time projection, and two640-parameter norms. The convolution weight/bias and norm numbers appear on their own actual cards. The total includes the time projection even though the selected five-edge connection claim does not prove its whole route; construction/shape authority is separate from connection authority.

## 3. First downsampler — ACCEPT the applied operation and counts; LIMITED as a connection-drawing trace

Occurrence down_blocks.0.downsamplers.0; canonical fact root.denoiser.spatial_mechanisms, claim_kind applied_function, proof selected_spatial_primitive_return_route.

Archived Downsample2D source (SHA2852f787..., lines97–147) selects Conv2d at stride2 and returns self.conv(hidden_states). Exact instance values have use_conv=True, norm=None and padding1, and the actual child carries the conv2d primitive witness with stride[2,2], kernel[3,3], channels320→320. This proves the stated selected stride-2 spatial operation. It does not justify guessing activation tensor dimensions.

The correct visible overview stage is instance_down_blocks__0. The occurrence's own fact-citing card displays Source-proven spatial primitive: torch.nn.Conv2d, Stride:2 and921,920 parameters. Independent arithmetic is320×320×3×3+320=921,920; the child card contains the actual convolution shapes.

However, this card uses constructed_children. Its actual drill SVG contains one convolution child and ZERO flow arrows. The source operation is correctly described, but there is no drawn input→convolution→output route. Zero reporter chain_gaps means the fact, card, visible stage, drill and numbers all link; it does not establish that a connected spatial mechanism was drawn.

For literal C8's three established-connection-to-drawing demonstrations, this third selection remains limited. The final sheet may accurately claim an applied-function/shape trace. It must not count this containment SVG as an additional routed-connection demonstration. Showing the already proved spatial route, or selecting an already drawn connection claim, would resolve that distinction without changing the accepted semantic fact.

## Reproduction and limits

From unfold-pkg:

    python3 verification/receipts/S8-final-three-claims-independent/replay_linkage.py
    python3 verification/receipts/S8-final-three-claims-independent/audit_semantics.py

Source-excerpts.json records exact hashes and relevant archived lines. semantic-results.json records qualified claim kinds, canonical stage/card identities, every selected subtree parameter shape, independent totals, actual SVG membership/arrow counts and artifact hashes. claim-traces.json preserves the corrected report's complete proof/archive references. The actual cards visibly retain execution investigation_missing/no_recipe_attempted; this review does not promote static evidence to execution evidence.

No product edits, model runs, pytest, output blessings or broader source-grammar audit. No new semantic false positive was established in these three selected facts. This is not acceptance of full model coverage, all C8 conditions, family differentials or final S8 completion.
