# Current approved gallery archive — 710da8e

Lossless archive of the 19 current galleries: all 294 actual PNGs, 19 diagram manifests, 19 completed Her Eyes reviews, report DTOs, source/image bindings, individual review records, generation logs, and aggregate indexes. The exact executed gallery script and input manifest are included separately under `executed/`.

`artifact-map.json` maps each complete original relative path to `streams/<original-path>.gzip`. Appending the suffix avoids collisions between original files that already have compressed or alternate extensions. Every stored stream was decompressed and compared byte for byte with its source; raw and stored sizes and SHA-256 hashes are recorded. Source galleries were not modified.

From a clean checkout, verify and optionally restore inspectable PNGs and Markdown with:

```sh
python3 verify_restore.py
python3 verify_restore.py --out /private/tmp/s8-current-gallery-restored
```

The destination must not exist. Restored gallery files are under its `gallery/` directory. Original source paths inside historical records remain unchanged; they are provenance, not a requirement for verification or restoration.

The historical generation results and `image-bindings.json` retain their pending-inspection state exactly as authored before review. Completed `her_eyes_review.md` and per-image review records supply the later review phase. The 18 non-UNet reviews cover 151 individually viewed PNGs; the independent SDXL review covers 143. Suggestions remain explicit and are not claimed as resolved.

These are native images from retained actual HTML, not new model parses or fresh Sable executions. Bloom's explicitly derived geometry report preserves the earlier actual report's semantic fields and the separately approved single attention-view signature delta. Original HTML and prior galleries remain in their existing historical receipts; this archive does not bless or replace them. Prior gallery streams are separately preserved in `S8-approved-prior-gallery-710da8e`.
