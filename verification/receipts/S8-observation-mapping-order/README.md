# Observation mapping order

The actual 9fe ordinary run exposed 253 block inventory deltas plus its signature because in-memory mapping insertion order differed after sorted-key JSON persistence. The report block walker now visits mapping keys canonically and retains list order exactly. Its shared observation function is used both by generation and read-back. Existing historical observations remain evidence of the failed check and are not rewritten.

Standalone probe_order.py passes keyed _call_6/_call_12 memory/JSON roundtrip and list roundtrip controls; reversing an actual list still changes both structure and signature. Actual page hashes, source hashes, semantic values and every other integrity check are unchanged. No model or pytest run, no blessing.
