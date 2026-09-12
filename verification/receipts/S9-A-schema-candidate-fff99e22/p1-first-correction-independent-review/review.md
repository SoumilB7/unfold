# P1 first correction, independent source review

RETURN for remaining same-object mutation paths. Exact requested position_fixed.py 200febb0… and test 0ec7d47a… were reviewed; ProgramIndex is byte-identical to the ninth frozen dependency. No tests/models were run.

Accepted bounded progress: finite builder/forward return chains refuse arbitrary zero_ methods and arithmetic; builder suffix and installer statement/call censuses refuse the ordinary alias, mutator and subscript-write cases. Forward checks refuse direct self.cache mutation/escape and preserve only the named read/wrapper operations. These close the initial examples.

Remaining concrete paths:

1. Forward `alias = self; alias.cache[...] = 0` before the original index_select return. The statement loop scans only explicit self.cache receivers, and the bare-self escape predicate runs only on call arguments. These two added assignments need no call, so the remaining call census stays valid. Direct `self.__dict__["cache"][...] = 0` likewise lacks a syntactic self.cache receiver. Refuse bare-self aliasing and namespace reflection in the exact statement/use census unless specifically proven.

2. Constructor `self.install(size,width); self.cache.zero_()` can destroy the installed table. The claim-origin checks require one direct install, reject prior returns and protected field assignments, but do not inspect the constructor's remaining calls. New installer/forward checks do not examine this constructor mutation. A finite constructor use guard must retain normal inherited initialization while refusing unsupported writes/mutators/escapes of the installed value or receiver.

Suggested cases are source-inferred falsifiers, not claimed executed reproductions. Preserve old unsupported DTO/value behavior as unqualified; do not silently alter product values. Helper and root have these exact cases. No output/installation approval is granted.
