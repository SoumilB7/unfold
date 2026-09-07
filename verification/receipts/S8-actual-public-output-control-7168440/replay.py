"""Compare actual44 public captures against reviewed pages without changing either."""
from pathlib import Path
import sys,json,gzip,copy,hashlib,dataclasses,re,time
ROOT=Path('/private/tmp/unfold-s8-final-verification-7168440');sys.path.insert(0,str(ROOT))
from model_unfolder.ir import ModelIR,EvidenceWarning
from model_unfolder.diagram import Diagram
from model_unfolder.block_schema import validate_block_tree,validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids
from model_unfolder.lint import lint_labels
from model_unfolder.sable import _projection_audit_findings
BASE=Path('/private/tmp/unfold-s8-typed-phase-7168440');PUBLIC=Path('/private/tmp/unfold-s8-coverage-7168440/actual-generate');OUT=Path('/private/tmp/unfold-s8-public-output-control-7168440');OUT.mkdir(exist_ok=False)
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):
 if dataclasses.is_dataclass(v):return canonical(dataclasses.asdict(v))
 if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
 if isinstance(v,(set,frozenset)):return sorted(canonical(x) for x in v)
 return v
def write(p,v):p.write_text(json.dumps(canonical(v),indent=2,sort_keys=True)+'\n')
def manifest():return {str(p.relative_to(ROOT)):sha(p) for d in ('model_unfolder','physics') for p in sorted((ROOT/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
def differences(a,b,path=()):
 if a==b:return []
 if isinstance(a,dict) and isinstance(b,dict):
  assert a.keys()==b.keys(),path
  return [d for k in a for d in differences(a[k],b[k],path+(k,))]
 if isinstance(a,list) and isinstance(b,list):
  assert len(a)==len(b),path
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+(i,))]
 return [{'path':list(path),'public':a,'reviewed':b}]
def lookup(v,path):
 for key in path:v=v[key]
 return v
rows=[]
for name,case in [('stable-diffusion-xl-base-1-0','sdxl-ordinary'),('sd-v1-4','sd14-ordinary')]:
 started=time.monotonic();out=OUT/case;out.mkdir();base=BASE/case;public=PUBLIC/name;before=manifest();write(out/'source-before.json',before)
 pins={str(p):sha(p) for p in (Path(__file__),base/'ir.json',base/'render-input.json',base/'page.html',base/'render-events.json.gz',public/'ir.json.gz',public/'display-ir.json.gz',public/'page.html.gz')};write(out/'input-pins.json',pins)
 raw=read(public/'ir.json.gz');display=read(public/'display-ir.json.gz');saved=read(base/'ir.json');capture=read(base/'render-input.json');delta=differences(raw,saved);assert len(delta)==(3 if case=='sdxl-ordinary' else 31)
 groups={tuple(d['path'][:-1]) for d in delta};lists=[]
 for path in sorted(groups,key=str):
  assert path[-3:-1]==('detail','fact_display_lines') and path[-1]=='root.denoiser.declared_constructor_defaults',path
  left=lookup(raw,path);right=lookup(saved,path);assert isinstance(left,list) and all(isinstance(v,str) for v in left+right)
  assert len(left)==len(right)==len(set(left))==len(set(right)) and set(left)==set(right)
  blockpath=path[:-3];block=lookup(raw,blockpath);oldblock=lookup(saved,blockpath);assert block['facts']==oldblock['facts']
  assert all(v in block['facts'] for v in left)
  lists.append({'path':list(path),'block_id':block['id'],'public_source_order':left,'reviewed_saved_fact_order':right,'unique_same_exact_members':True,'actual_card_facts_order_unchanged':True})
 write(out/'ir-deltas.json',{'scalar_differences':delta,'metadata_lists':lists})
 # These are actual captured public bytes. Mount substitution is a diagnostic only.
 actual=gzip.decompress((public/'page.html.gz').read_bytes()).decode();savedpage=(base/'page.html').read_text();mounts=set(re.findall(r'uf-[a-z0-9]{10}',actual));assert len(mounts)==1;publicmount=next(iter(mounts));savedmount=capture['mount_id'];assert actual.replace(publicmount,savedmount)==savedpage
 ir=ModelIR(**copy.deepcopy(raw));ir.warnings=[EvidenceWarning(w['check'],w['summary'],tuple(w['details']),w['detail']) if w['kind']=='evidence' else w['text'] for w in capture['warnings']];assert ir.to_dict()==read(public/'ir.json.gz')
 d=Diagram(ir);d._mount_id=savedmount;assert canonical(d.param_count())==capture['parameters'];assert d.to_ir()==display
 page=d.to_html();(out/'replayed-page.html').write_text(page);assert page==savedpage;assert page.replace(savedmount,publicmount)==actual
 events=(json.dumps(canonical(d.render_events()),sort_keys=True,separators=(',',':'))+'\n').encode();(out/'replayed-events.json.gz').write_bytes(gzip.compress(events,mtime=0));assert events==gzip.decompress((base/'render-events.json.gz').read_bytes())
 write(out/'replayed-render-input.json',{'ir':d.to_ir(),'parameters':canonical(d.param_count()),'warnings':capture['warnings'],'mount_id':savedmount})
 checks={f.__name__:canonical(f(page)) for f in (validate_click_coupling,validate_no_dotted_arrows,validate_no_dotted_boundaries,validate_unique_ref_ids)};checks['validate_block_tree']=canonical(validate_block_tree(ir));checks['label_lint']=lint_labels(d.to_ir());checks['projection_audit']=_projection_audit_findings(d.to_ir(),d.render_events());assert all(not v for v in checks.values())
 after=manifest();write(out/'source-after.json',after);assert before==after;assert pins=={p:sha(Path(p)) for p in pins}
 result={'status':'PASS','case':case,'public_model':name,'production_commit':'7168440c81a3e79b1e220437e2f9fa77e25bc296','public_capture_dir':str(public),'reviewed_case_dir':str(base),'ir_scalar_difference_count':len(delta),'metadata_permutation_list_count':len(lists),'all_ir_differences_exact_unique_string_permutations':True,'actual_card_facts_order_unchanged':True,'public_mount':publicmount,'reviewed_mount':savedmount,'public_mount_occurrences':actual.count(publicmount),'actual_public_html_only_exact_mount_token_difference':True,'actual_public_display_ir_exact_replay_input':True,'replayed_html_bytes_exact_reviewed_page':True,'replayed_event_bytes_exact_reviewed_events':True,'public_html_uncompressed_sha256':hashlib.sha256(actual.encode()).hexdigest(),'reviewed_html_sha256':sha(base/'page.html'),'replayed_html_sha256':sha(out/'replayed-page.html'),'replayed_events_uncompressed_sha256':hashlib.sha256(events).hexdigest(),'parameters_and_warning_metadata_preserved':True,'checks':checks,'elapsed_seconds':round(time.monotonic()-started,3),'no_model_or_pytest':True,'historical_six_condition_raw_identity_untouched':True,'blessed':False};write(out/'result.json',result);rows.append(result);print(json.dumps(result),flush=True)
write(OUT/'summary.json',rows)
