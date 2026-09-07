# Final offline wheel smoke

PASS at 5227e9c. Exact git archive inputs were built offline without dependencies or build isolation, installed separately, and imported from outside the checkout. UNet cutover, call/iteration bindings, retained-index context and all physics modules resolve exclusively to the installed wheel. Logs and command/input/output hashes are retained. Host dependencies were reused; this is not a clean environment model run, broad gate, or release approval.
