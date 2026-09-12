"""Reproduce the frozen f664467 loader flag mismatch without constructing a model."""
from pathlib import Path
import sys,tempfile,types,hashlib,json,__future__
sys.path.insert(0,'/private/tmp/unfold-s8-final-campaign')
from physics.source_override import SourceOverride,_ExactSourceLoader
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'source_fixture.py'
    raw=b'def forward(value):\n    return value\n'
    path.write_bytes(raw)
    module=types.ModuleType('probe.source_fixture')
    loader=_ExactSourceLoader(SourceOverride(module.__name__,str(path),hashlib.sha256(raw).hexdigest()),set())
    loader.exec_module(module)
    expected=next(item for item in compile(raw,str(path),'exec',dont_inherit=True).co_consts if isinstance(item,types.CodeType))
    actual=module.forward.__code__
    result={'actual_flags':actual.co_flags,'source_compiled_flags':expected.co_flags,
            'xor':actual.co_flags^expected.co_flags,'future_annotations_flag':__future__.annotations.compiler_flag,
            'codes_equal':actual==expected,'instruction_bytes_equal':actual.co_code==expected.co_code}
    print(json.dumps(result,indent=2))
    assert not result['codes_equal'] and result['xor']==result['future_annotations_flag']
