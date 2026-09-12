# Lossless full-IR archive recovery

Exactly eight full actual example IR streams had collided with their existing canonical structural ir.json.gz files during packaging. All eight raw campaign streams still matched their original recorded SHA and byte count. They are now stored separately as ir.full.json.gz. Existing canonical ir.json.gz files are byte-unchanged.

The exact current pre-correction map is preserved as artifact-map.pre-full-ir-recovery.json; the earlier executed map remains artifact-map.executed-original.json. Only eight stored paths and storage hashes changed. Current script-archive map corrections and all logical original hashes/byte counts remain intact.

Full verification passes:1647unique destinations,1647stored SHA checks,1647decoded/original SHA and byte-count checks. The other1639entries are unchanged. This repairs archival storage only; no production, tests, gates, actual captures, renders or baselines changed.
