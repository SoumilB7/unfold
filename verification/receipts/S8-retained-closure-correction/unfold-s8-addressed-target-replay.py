import dataclasses,gzip,hashlib,importlib.util,json,sys,time
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]);out=Path(sys.argv[2]);slug=sys.argv[3];out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('address_replay_generator',root/'scripts/generate_s7_shadow.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
from model_unfolder.evidence.execution_recipe import RecipeAttemptBundle,RecipeResolution
from physics.relation_observation import RelationObservationResult

def read(p):return json.loads(gzip.decompress(p.read_bytes()))
def write(name,value):
 p=out/name;data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();p.write_bytes(gzip.compress(data,mtime=0) if p.suffix=='.gz' else data)
def canonical(v):return json.loads(json.dumps(v))
def sources():return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for area in ('model_unfolder','physics','scripts') for p in (root/area).rglob('*') if p.suffix in ('.py','.yaml','.yml')}
before=sources();write('source-before.json',before)
old=Path('/private/tmp/unfold-s8-final-verification-96d0c1e/verification/s7');observations=read(old/'observations'/f'{slug}.json.gz');relations=read(old/'relations'/f'{slug}.json.gz');original=read(old/'models'/f'{slug}.json.gz')
saved_relations=[RelationObservationResult.from_dict(r) for r in relations['results']];used=[]
def signature(request,resolution):
 write('recomputed-resolution.json',dataclasses.asdict(resolution))
 assert resolution==RecipeResolution.from_dict(observations['resolution']),g._payload_differences(canonical(dataclasses.asdict(resolution)),observations['resolution'])
 assert request.to_dict()==g._s6_request(slug,original['inventory_provenance']['config_sha256']).to_dict()
 return RecipeAttemptBundle.from_dict(observations)
def relation(request,recipe,stack):
 matches=[(i,r) for i,r in enumerate(saved_relations) if r.recipe==recipe and r.stack_path==stack]
 assert len(matches)==1 and matches[0][0] not in used
 used.append(matches[0][0]);return matches[0][1]
def forbid(*a,**kw):raise AssertionError('no fresh runtime/build in saved address replay')
real_sources=g._source_inputs
def capture(*args):
 result=real_sources(*args);write('ir.json.gz',result[4].to_dict());write('facts.json.gz',result[0].facts.to_dict());return result
start=time.monotonic()
with patch.object(g,'_source_inputs',capture),patch.object(g,'_run_signature_recipe',signature),patch.object(g,'observe_relations_in_subprocess',relation),patch.object(g,'inventory_in_subprocess',forbid):
 artifact,obs,rels=g._one(next(t for t in g._targets() if t['slug']==slug),write_relations=False,output=old)
assert canonical(obs)==observations and canonical(rels)==relations
assert len(used)==len(saved_relations) and sources()==before
write('model.json.gz',artifact);write('observation.json.gz',obs);write('relations.json.gz',rels)
deltas=g._payload_differences(canonical(artifact),original,limit=100000)
write('summary.json',{'slug':slug,'elapsed_seconds':round(time.monotonic()-start,3),'model_equal_original':not deltas,'delta_paths':deltas,'observations_exact':True,'relations_exact':True,'source_unchanged':True,'no_new_model_or_execution':True})
write('source-after.json',sources());print('complete',slug,'deltas',len(deltas),flush=True)
