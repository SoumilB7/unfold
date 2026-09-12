# Final offline wheel smoke

PASS at96d0c1e. Build inputs are exact git archive bytes for pyproject/README and both production packages. Built offline with no dependency resolution or build isolation, installed to a separate target, then imported installed UNet cutover/call/iteration modules and every physics module from outside the checkout. Exact import paths, archive/wheel hashes and logs are retained. Host dependencies were reused; this is not a clean environment model run, full gate or release approval.
