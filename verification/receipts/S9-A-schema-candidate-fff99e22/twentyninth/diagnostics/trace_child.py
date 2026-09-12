import json,sys
from pathlib import Path
rows=[]
def trace(frame,event,arg):
 if not frame.f_code.co_filename.endswith('/evidence/class_default_value.py'): return trace
 if event=='return' and (arg is None or arg is False):
  rows.append({'function':frame.f_code.co_name,'line':frame.f_lineno,'returned':arg,'locals':{k:str(v)[:1800] for k,v in frame.f_locals.items() if k in {'config_path','operand_path','prefix','relative','binding','candidate','alias','callee','parent','allowed','owner_symbol','parameter','stored_field','init','node','plans','child_sites','expression'}}})
 return trace
sys.settrace(trace)
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','tests/test_s9_child_class_default.py::test_exact_constructed_child_literal_is_qualified_without_root_borrowing[False]','tests/test_expanded_json.py::test_expanded_json_completes_qwen2_audio_sparse_text_config'])
sys.settrace(None)
Path('/private/tmp/unfold-s9a-twentyninth/child-trace/returns.json').write_text(json.dumps(rows,indent=2)+'\n')
sys.exit(code)
