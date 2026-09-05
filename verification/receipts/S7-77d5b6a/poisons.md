# S7 v2.6.2 closure poison index

The prior S7 poison set remains active. This closure adds and verifies:

1. An existence/constructor declaration offered for a connection claim raises.
2. A config-only fact offered for an applied-function claim raises.
3. A projected fact without a reader claim-kind declaration raises.
4. Every bridge declaration names a fact in the canonical registry.
5. Every persisted projected fact has exactly one closed claim kind.
6. Every target has one persisted recipe decision and at most one retry.
7. A non-dtype failure cannot trigger the retry.
8. A retry that changes checkpoint dtype—or anything except the exact derived
    execution dtype and retry citation—is rejected.
9. A failed recipe resolution cannot carry a runtime execution result.
10. Matrix recipe status, resolution, dtype, checkpoint provenance and retry
    count must equal the per-target observation bundle.
11. Direct checkpoint bf16 and null-checkpoint-then-retry examples remain
    separately persisted and testable.
