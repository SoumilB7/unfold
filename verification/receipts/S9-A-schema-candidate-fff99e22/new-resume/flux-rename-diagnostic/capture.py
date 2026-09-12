"""Root-only diagnostic of the unchanged FLUX name-blind two-pass parse.

Observation wrappers delegate unchanged and re-raise exceptions. No comparison
fields are suppressed beyond identity_guard's EXISTING normalization.
"""
import argparse
from pathlib import Path
import dataclasses
import gzip
import hashlib
import json
import sys
import traceback

args = argparse.ArgumentParser()
args.add_argument('--output', required=True, type=Path)
args = args.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(Path.cwd()))
from test_support import FLUX
from model_unfolder.adapters import find_adapter
from model_unfolder.parser import _coerce
from model_unfolder.evidence import context as contexts
from model_unfolder.evidence import config_access
from model_unfolder.evidence import identity_guard
from model_unfolder.adapters.transformer import parser as transformer

phase='setup';captures=[];default_attempts=[]
def write(name,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
 path=out/name
 path.write_bytes(gzip.compress(data,mtime=0) if name.endswith('.gz') else data)
def prepared_record(p):
 if p is None:return None
 return {'document':p.document,'checkpoint':p.checkpoint,'class_overlay':p.class_overlay,
         'provenance':p.provenance,'failure':p.failure.to_dict() if p.failure else None}
def bundle_record(b):
 return {key:getattr(b,key,None) for key in ('component_model_types','component_architectures','component_files','supporting_files')}
original_parse=transformer.parse
original_premise=config_access._class_default_premise

def observe_premise(cfg,path,spellings,value,component):
 result=original_premise(cfg,path,spellings,value,component)
 p=config_access.current_prepared_document.get()
 default_attempts.append({'phase':phase,'component':component,'path':list(path),
  'spellings':list(spellings),'value':value,'premise_retained':result is not None,
  'prepared_failure':p.failure.to_dict() if p and p.failure else None,
  'prepared_overlay_has_path':bool(p is not None and '.'.join(path) in p.class_overlay),
  'prepared_overlay_value':p.class_overlay.get('.'.join(path)) if p else None})
 return result

def observe_parse(cfg,context=None):
 p=config_access.current_prepared_document.get();row={'phase':phase,'namespace':getattr(context,'component_namespace',None),'prepared':prepared_record(p),'source_bundle':bundle_record(context.source_bundle),'context_class_defaults':context.class_defaults,'context_defaults_by_path':[{'path':list(k),'values':v} for k,v in context.class_defaults_by_path.items()]};captures.append(row)
 try:
  ir=original_parse(cfg,context=context)
  row['ir']=ir.to_dict();row['facts']=context.facts.to_dict();row['status']='ok'
  return ir
 except Exception as exc:
  row['status']='exception';row['exception']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
 finally:
  row['events']=[{f.name:getattr(e,f.name) for f in dataclasses.fields(e) if f.name!='document_token'} for e in context.config_access.events]

transformer.parse=observe_parse;config_access._class_default_premise=observe_premise
cfg=_coerce(FLUX);context=contexts.ParseContext.build(cfg,source='local');adapter=find_adapter(cfg)
write('input.json',cfg);write('root-address.json',{'source_bundle':bundle_record(context.source_bundle),'defaults_by_path':[{'path':list(k),'values':v} for k,v in context.class_defaults_by_path.items()]})
token=contexts.active_parse_context.set(context)
try:
 phase='original';original_ir=adapter.parse(cfg,context=context)
 phase='scrubbed';scrubbed_ir=adapter.parse(identity_guard.scrub_semantic_identity(cfg),context=context)
 original=identity_guard._normalized_structure(original_ir.to_dict());scrubbed=identity_guard._normalized_structure(scrubbed_ir.to_dict())
 write('original-ir.json.gz',original_ir.to_dict());write('scrubbed-ir.json.gz',scrubbed_ir.to_dict())
 write('original-existing-normalization.json.gz',original);write('scrubbed-existing-normalization.json.gz',scrubbed)
 result=identity_guard.NameBlindResult(original==scrubbed,original,scrubbed)
 write('result.json',{'structural_equal':result.structural_equal,'changed_paths':list(result.changed_paths),'scope':'Diagnostic wrappers; existing normalization only, no acceptance'})
finally:
 contexts.active_parse_context.reset(token);transformer.parse=original_parse;config_access._class_default_premise=original_premise
 write('children.json.gz',captures);write('default-premise-attempts.json.gz',default_attempts)
