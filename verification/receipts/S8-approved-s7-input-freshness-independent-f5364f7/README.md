# Approved corpus metadata and S7 freshness

Independent read-only inspection confirms exactly 19 changed entries in matrix.sources versus `_source_hashes(_targets())`. Every entry is one of the 19 owner-approved re-blessed corpus JSON files. No production Python, physics, YAML, pilot or other target input hash changed.

Each prior whole-file hash was recovered from the exact preserved pre-bless fixture. Each new whole-file hash matches the current file. The model/config tuples returned by the existing `_read_payload` are exactly equal, and config/source/model fields are unchanged. Generator lines 162–172 and 476–477 consume these model/config values; review signatures and gallery metadata are not architectural inputs. The conservative whole-file freshness guard correctly became stale after their metadata changed.

The lawful correction is an explicitly derived metadata update of only those 19 matrix.sources digests. Keep all 117 artifact files, all 39 model summaries and targets, denominator, claim tables and logical artifact hashes exact. Do not change the guard or claim a new model/physics run. The original matrix, targets and 117 payload hashes are retained here, alongside the exact source-map diff. Final corrected bytes require independent comparison before this proposal becomes acceptance.

No model, renderer, pytest or production edit ran during this review. The frozen f5364f7 gate remains an actual failure and must retain that verdict.
