from pathlib import Path
import sys,json,dataclasses
sys.path.insert(0,str(Path.cwd()))
from model_unfolder import config_to_ir
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.config_access import bound_document
cfg=json.loads(Path('tests/sable_test_corpus/llama-7b.json').read_text())['config']
context=ParseContext.build(cfg);ir=config_to_ir(cfg,parse_context=context)
rows=[]
with bound_document(context.prepared_documents['root']):
 for (name,path),result in context.reader_results.items():
  if name not in ('decoder.ffn.mechanism','decoder.attention.score_scaling'):continue
  w=result.claim_witness
  row={'name':name,'path':path,'status':result.status,'witness':type(w).__name__,'activation_gap':getattr(w,'activation_gap',None)}
  if w:
   key='activation' if 'ffn' in name else 'scores_scale';owner='decoder.ffn' if 'ffn' in name else 'decoder.attention'
   try:row['projection']=str(w.project(owner,key,context.prepared_documents['root'].prepared))
   except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
  rows.append(row)
Path('/private/tmp/unfold-s9a-eleventh/actual-llama-reader-reasons.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
