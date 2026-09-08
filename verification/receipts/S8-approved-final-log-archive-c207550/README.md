# Historical raw-log archival

The final staged static check rejected trailing whitespace in five exact historical pytest tracebacks. Each original stream is now stored as appended `.log.gz`, with byte/hash/length equality verified after decompression and an original-logical-path map. Original review hashes and verdicts remain historical and unchanged. Original full-gate logs are also retained in the separate lossless full-run archive. No content stripping, checker exception, source/test change or gate rerun occurred. The original static return is retained losslessly.
