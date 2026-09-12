"""Read-only actual Graph/Region capture and same-IR rendering diagnostic."""
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
import ast,copy,dataclasses,difflib,gzip,hashlib,importlib.util,json,subprocess,sys

def canonical(v):
 if dataclasses.is_dataclass(v):return canonical(dataclasses.asdict(v))
 if isinstance(v,dict):return {str(k):canonical(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [canonical(x) for x in v]
 if isinstance(v,(set,frozenset)):return sorted(canonical(x) for x in v)
 return v

def write(p,v):p.write_text(json.dumps(canonical(v),indent=2,sort_keys=True,default=str)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manifest(root):return {str(p.relative_to(root)):sha(p) for d in ('model_unfolder','physics','scripts') for p in sorted((root/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}

def distinct(page):
 from model_unfolder.preview import svg_views,_visual_hash
 rows=[];seen=set()
 for label,svg in svg_views(page):
  h=_visual_hash(svg)
  if h not in seen:rows.append((label,h,svg));seen.add(h)
 return rows

class GraphCapture:
 def __init__(self):self.records=[];self.phase='actual';self.regions={}
 def __enter__(self):
  from model_unfolder.renderers.html import graph_engine,op_render
  from model_unfolder.preview import _visual_hash
  self.stack=ExitStack();original_region=op_render.region_to_graph;original_graph=graph_engine._render_graph_in_context
  def region_capture(region,*args,**kwargs):
   graph=original_region(region,*args,**kwargs);self.regions[id(graph)]=(graph,copy.deepcopy(region));return graph
  def graph_capture(graph,*args,**kwargs):
   svg=original_graph(graph,*args,**kwargs);region=self.regions.get(id(graph),(None,None))[1];edges=[]
   for parallel in graph.parallels:
    for lane in parallel.norm_lanes():
     if lane.ids:continue
     source=lane.src or parallel.src
     for target in lane.dst or [parallel.dst]:edges.append({'source':source,'target':target,'canonical_region_edge':bool(region is not None and any(e.src==source and e.dst==target for e in region.edges))})
   self.records.append({'phase':self.phase,'view':args[2],'graph':canonical(graph),'region':canonical(region),'empty_lane_edges':edges,'svg_sha256':hashlib.sha256(svg.encode()).hexdigest(),'visual_hash':_visual_hash(svg),'block_path':[b.get('id') for b in kwargs['context'].block_stack]});return svg
  for module in list(sys.modules.values()):
   if getattr(module,'__name__','').startswith('model_unfolder.renderers.html') and getattr(module,'region_to_graph',None) is original_region:self.stack.enter_context(patch.object(module,'region_to_graph',region_capture))
  self.stack.enter_context(patch.object(graph_engine,'_render_graph_in_context',graph_capture));return self
 def __exit__(self,*args):return self.stack.__exit__(*args)

def surfaces(diagram,page,root,slug,out):
 from test_support import preservation
 from model_unfolder.params import estimate_params
 assert Path(preservation.__file__).resolve().is_relative_to(root)
 structural,ledger=preservation.split_structural_ir(diagram.to_ir());docs={'ir':structural,'ledgers':ledger,'expanded':diagram.to_json(),'params':estimate_params(diagram.ir),'html_meta':preservation.html_meta(page)};expected=json.loads((root/'tests/preservation_expected_manifest.json').read_text())['witnesses'][slug];hashes={}
 for name,value in docs.items():
  payload=preservation._canon_bytes(value);(out/(name+'.json.gz')).write_bytes(gzip.compress(payload,mtime=0));hashes[name]={'actual':hashlib.sha256(payload).hexdigest(),'expected':expected['surfaces'][name]}
 write(out/'surface-hashes.json',hashes);write(out/'comparator-pin.json',{'source':preservation.__file__,'sha256':sha(preservation.__file__),'input_sha256':preservation.input_sha256(root/'tests/sable_test_corpus'/f'{slug}.json')});return hashes

def helper_equivalence(old_source,current_source):
 """Pin the old function's transitive module-level dependencies, not semantics."""
 def bindings(source):
  result={}
  for node in ast.parse(source).body:
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):result[node.name]=node
   elif isinstance(node,(ast.Import,ast.ImportFrom)):
    for alias in node.names:
     name=alias.asname or (alias.name.split('.')[0] if isinstance(node,ast.Import) else alias.name)
     result[name]=ast.Import(names=[alias]) if isinstance(node,ast.Import) else ast.ImportFrom(module=node.module,names=[alias],level=node.level)
   elif isinstance(node,ast.Assign):
    for target in node.targets:
     if isinstance(target,ast.Name):result[target.id]=node
   elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):result[node.target.id]=node
  return result
 old=bindings(old_source);new=bindings(current_source);pending=[n.id for n in ast.walk(old['_draw_parallel']) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)];seen=set();rows=[]
 while pending:
  name=pending.pop()
  if name in seen or name=='_draw_parallel' or name not in old:continue
  seen.add(name);left=ast.dump(old[name],include_attributes=False);right=ast.dump(new[name],include_attributes=False) if name in new else None
  rows.append({'binding':name,'kind':type(old[name]).__name__,'old_ast_sha256':hashlib.sha256(left.encode()).hexdigest(),'current_ast_sha256':hashlib.sha256(right.encode()).hexdigest() if right else None,'exact_ast_equal':left==right})
  pending.extend(n.id for n in ast.walk(old[name]) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))
 return {'scope':'Transitive same-module functions/classes/constants/import bindings referenced by accepted _draw_parallel; import names are compared individually. Source AST identity is a diagnostic compatibility check, not new model-mechanism evidence.','bindings':sorted(rows,key=lambda r:r['binding']),'all_exact':all(r['exact_ast_equal'] for r in rows)}

def old_parallel(diagram,page,root,slug,out,observer):
 from model_unfolder.diagram import Diagram
 from model_unfolder.renderers.html import graph_engine
 from test_support import preservation
 out.mkdir();source=subprocess.check_output(['git','show','83140f1:model_unfolder/renderers/html/graph_engine.py'],cwd=root);equivalence=helper_equivalence(source,(root/'model_unfolder/renderers/html/graph_engine.py').read_bytes());write(out/'helper-equivalence.json',equivalence);assert equivalence['all_exact'],'Old renderer helper dependencies differ; cannot attribute old-function-only';p=out/'graph_engine_83140.py';p.write_bytes(source);name='model_unfolder.renderers.html._old_parallel_capture';spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 actual_records=[r for r in observer.records if r['phase']=='actual:'+slug];assert all(par['src'] is not None for r in actual_records for par in r['graph']['parallels']),'independent-input graph needs separate diagnostic'
 d=Diagram(copy.deepcopy(diagram.ir));d._mount_id=diagram._mount_id;assert d.to_ir()==diagram.to_ir();observer.phase='old_parallel:'+slug
 with patch.object(graph_engine,'_draw_parallel',module._draw_parallel):oldpage=d.to_html()
 (out/'page.html').write_text(oldpage);write(out/'render-events.json',d.render_events());current=distinct(page);old=distinct(oldpage);expected=json.loads((root/'tests/preservation_expected_manifest.json').read_text())['witnesses'][slug];oldviews=[{'label':label,'sha256':h} for label,h,_ in old];newviews=[{'label':label,'sha256':h} for label,h,_ in current]
 views=out/'views';views.mkdir()
 for prefix,rows in [('old',old),('current',current)]:
  for n,(label,h,svg) in enumerate(rows):(views/f'{prefix}-{n:03d}.svg').write_text(svg)
 changed=[]
 for n in range(max(len(current),len(old))):
  c=current[n] if n<len(current) else None;o=old[n] if n<len(old) else None
  if c is None or o is None or c[:2]!=o[:2]:
   changed.append({'index':n,'old':o[:2] if o else None,'current':c[:2] if c else None})
   if c and o:(views/f'{n:03d}.diff.gz').write_bytes(gzip.compress(''.join(difflib.unified_diff(o[2].splitlines(True),c[2].splitlines(True),fromfile='old-parallel-only',tofile='current')).encode(),mtime=0))
 result={'slug':slug,'same_ir':True,'same_mount':True,'no_second_parse':True,'old_views':oldviews,'current_views':newviews,'expected_views':expected['views'],'old_locked_views_recovered':oldviews==expected['views'],'old_html_meta_hash':hashlib.sha256(preservation._canon_bytes(preservation.html_meta(oldpage))).hexdigest(),'expected_html_meta_hash':expected['surfaces']['html_meta'],'changed_views':changed,'cause_review':'Actual canonical edges and per-output native geometry still require review; hash recovery alone is not approval.','blessed':False};write(out/'result.json',result);return result
