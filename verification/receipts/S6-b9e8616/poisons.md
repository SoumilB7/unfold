# S6 poison index

All poisons are permanent tests in `tests/test_s6_physics.py` and were green in
the focused committed-tree lane. “Green” means the injected violation was
rejected or retained as a typed unresolved/failure result.

1. Constructor opens a socket → `NetworkRefused`, with no destination echo.
2. Constructor launches a download client → `NetworkRefused` before spawn.
3. Constructor exceeds its wall limit → `TimeoutExpired`; process group killed.
4. Constructor touches memory beyond the process-tree RSS ceiling →
   `MemoryLimitExceeded`.
5. Constructor writes more than an OS pipe buffer → no deadlock; captured tail
   remains bounded to 65,536 characters.
6. A first-call module appears after construction → `lazy_observed` with its
   exact path and runtime class.
7. Tensor-value control flow under FakeTensor → `ExecutionUnresolved`, never a
   guessed branch.
8. A false recipe library version → typed failure before observation.
9. Nested mapping/tuple recipe arguments → exact structure survives assembly.
10. Constructor mutates its nested config input → checkpoint config hash and
    caller document remain unchanged.
11. Unconditional `None` assignment → excluded from guarded-child evidence;
    conditional `None` retains predicate and branch.
12. Forged DTO status/kind/index/hash/class combinations → constructor error.
13. Missing pilot, request, inventory, observation, or manifest row → gate red.
14. Pilot source-file/config/artifact hash mismatch → gate red.
15. SD3.5 block-0 counts differing from 1 SDPA / 16 add / 16 mul → gate red.
16. PixArt factory remap absent or SD3.5 guarded `attn2=None` unproven → gate red.
17. Any production import of `physics`, or model identity in the generic
    substrate → gate red.
