# Conditioning expression port closure

Actual b44dd25 primary drill retained the temb call port but stopped at the indexed conditional expression `emb + aug_emb if aug_emb is not None else emb`. The bounded reader now exposes both source operand routes with an unresolved guard, preserving the addition boundary and the false alternative. It does not interpret helper bodies, condition truth or operator dispatch. Named writes in an expression guard refuse old-operand authority. Supported conditional expressions no longer become fabricated intervening local-rebinding statements solely because their operands contain calls.

43 primary/route tests passed in 0.16 seconds, including the actual source-to-proof-to-ledger/citation regression and conditional-expression/named-write poisons. A bounded actual SDXL source proof preflight accepted the ledger/citation with 197 canonical refs and all three down-stage temb ports retained as conditional routes. This preflight did not rebuild the model; a fresh actual page is still required.

The neutral false-alternative label was already present in3693545 and was not changed here. Candidate3788262 was never run; the helper paused before launch. No outputs blessed or pushed.
