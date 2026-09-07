# R1 owner storage-write controls — RETURN

Standalone source-index probes on the working correction reader, based on d3a4d61 plus its pending R1 edits. This is not an immutable whole-tree gate. Exact reader bytes and SHA-256 are retained; the reader stayed unchanged during the run.

The unmodified control retains the first-to-second call connection. Explicit writes through self._modules and self.__dict__, and deletion of self.second, incorrectly retain it too. Existing source observations contain these statements; a bounded reader must account for their effect or refuse the binding claim.

Command: `PYTHONDONTWRITEBYTECODE=1 python3 verification/receipts/S8-authority-followup/owner-storage-writes/reproduce.py`. Exit 1; three failed authority controls, one positive control passed. No pytest, real model execution, production edit or blessing.
