# S8 legacy comparison halves — frozen 4637ef4

Serial explicit `--condition legacy` runs from detached `4637ef4b77a8139326a8643b9f7eb68ab9d0f93c`, using the exact raw inputs from the successful frozen family verification. Both exited zero with no wiring problems and unchanged implementation source manifest. SDXL cold 6.443 seconds; SD-v1-4 cold 5.321 seconds.

These are the OLD-path halves for later differential review. They are not blessed, do not prove new-path preservation, and do not qualify legacy facts. Exact HTML bytes are deterministically compressed; IR/facts/observations/results, source records, input, logs and cold receipts are retained per witness. Actual uncompressed scratch HTML remains under `/private/tmp/unfold-s8-legacy-witnesses/<witness>/legacy/page.html`. No six-condition new-path runs or pytest were performed.
