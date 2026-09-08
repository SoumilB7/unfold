# FD diagnostic v1 and preceding public-v6 failure

This receipt preserves two distinct actual outcomes. The public cold-v6 attempt retains its original assertion failure and empty result cache. The later FD diagnostic returned an inventory exactly equal to its comparison inventory, but used a modified worker command and an additional `fd_worker.py`; it is diagnostic evidence, not a normal public cache or latency result.

The actual dependency sidecar remains **ineligible**, with both `unresolved_file_descriptor_input` and the extra diagnostic worker's external-read refusal. No FD exemption or broad suppression was introduced by this observation. The diagnostic records integer-open event addresses/modes/flags without interpreting an absent recurrence as proof of an earlier failure's cause.

The exact diagnostic request, inventory, dependency sidecar, integer-open events, result, adjacent log, preparation scripts/source manifest, and independent source/actual reviews are retained. The preparation/review history includes the pre-use selector correction. The separate public-v6 result/IR/inventory/pins/log and its exact preparation/retarget files remain under their original logical namespaces. `public-entries-v6` is explicitly empty.

`file-list.txt` enumerates all 26 original streams. The appended-`.gzip` map is injective; all 57,349,987 original bytes were restored to separate files and checked by SHA-256 and size, with full directory/file membership and unchanged original-byte checks. Map SHA-256: `2d230018d6098eeb76b9bb573a3f9d6920b67c98958877d0a206fc69f130a24d`.

Packaging ran no diagnostic code, model, test, or renderer. It made no baseline change, normal-worker equivalence claim, latency median, or historical-cause inference. External files mentioned inside manifests were not imported into the archive.
