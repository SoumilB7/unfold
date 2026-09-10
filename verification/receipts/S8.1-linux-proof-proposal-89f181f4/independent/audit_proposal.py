from pathlib import Path
import json,gzip,hashlib,re,ast,collections
from collections.abc import Mapping
r=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg'); out=Path('/private/tmp/unfold-s81-portable-proof-verification'); p=out/'sdxl'; proposal=p/'proposal'; original=r/'verification/s7'; review=Path('/private/tmp/unfold-s81-portable-proof-independent-review')
h=lambda b:hashlib.sha256(b).hexdigest(); read=lambda p:json.loads(gzip.decompress(p.read_bytes()))
tree=ast.parse((r/'scripts/generate_s7_shadow.py').read_text()); keep=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_SEMANTIC_ENVIRONMENT_PATHS','_SEMANTIC_DIAGNOSTIC_PATHS'} for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='_semantic_payload')];ns={'Any':object,'Mapping':Mapping};exec(compile(ast.Module(body=keep,type_ignores=[]),'<unchanged semantic function>','exec'),ns);semantic=ns['_semantic_payload']
def diff(a,b,path='$'):
 assert type(a)==type(b),(path,'type')
 if isinstance(a,dict):
  assert a.keys()==b.keys(),(path,'keys')
  return [v for k in sorted(a) for v in diff(a[k],b[k],path+'.'+k)]
 if isinstance(a,list):
  assert len(a)==len(b),(path,'length')
  return [v for i,(x,y) in enumerate(zip(a,b)) for v in diff(x,y,path+f'[{i}]')]
 return [] if a==b else [{'path':path,'before':a,'after':b}]
relative='models/stable-diffusion-xl-base-1-0.json.gz';before=read(original/relative);candidate=read(proposal/relative);actual=read(p/'actual-model.json.gz');delta=diff(before,candidate)
assert delta==json.loads((p/'full-semantic-delta.json').read_text()) and len(delta)==200
assert semantic(candidate)==semantic(actual)
pattern=re.compile(r'^\$\.table\.occurrences\[\d+\]\.projection\.fact_claim_proofs\[\d+\]\.index_fingerprints\[0\]$');assert all(pattern.fullmatch(v['path']) for v in delta)
counts=collections.Counter((v['before'],v['after']) for v in delta);assert sorted(counts.values())==[8,192]
for kind,singular in [('observations','observation'),('relations','relations')]:
 assert semantic(read(original/kind/'stable-diffusion-xl-base-1-0.json.gz'))==semantic(read(p/f'actual-{singular}.json.gz'))
mb=json.loads((original/'matrix.json').read_text());mc=json.loads((proposal/'matrix.json').read_text());md=diff(mb,mc);assert md==json.loads((p/'matrix-delta.json').read_text())
assert all(v['path'].startswith('$.sources.') or v['path'] in {'$.artifacts.'+relative,'$.logical_artifacts.'+relative} for v in md);assert mb['models']==mc['models']
report=json.loads((p/'result.json').read_text());renewed=[];unchanged=0
for a in sorted(original.rglob('*')):
 if not a.is_file():continue
 rel=a.relative_to(original).as_posix();b=proposal/rel;assert b.is_file();ah,bh=h(a.read_bytes()),h(b.read_bytes())
 if ah!=bh:renewed.append({'path':rel,'before_sha256':ah,'after_sha256':bh})
 elif rel.startswith(('models/','observations/','relations/')) and rel.endswith('.json.gz'):unchanged+=1
assert renewed==report['renewed'] and unchanged==116
result=json.loads((out/'result.json').read_text());assert result['candidate_pins_unchanged'] and result['error'] is None
for v in result['lanes'].values():assert v['passed'] and v['returncode']==0 and v['before']==v['after'] and v['artifacts_before']==v['artifacts_after']
assert '79 passed' in (out/'portable-proof-controls.log').read_text()
record={'decision':'ACCEPT_ACTUAL_TWO_FILE_PROPOSAL_FOR_OWNER_REVIEW','reviewer':'independent_review','base_commit':result['base_commit'],'candidate_manifest_sha256':result['manifest_sha256'],'actual_result_sha256':h((out/'result.json').read_bytes()),'proposal_result_sha256':h((p/'result.json').read_bytes()),'approved_for_review_paths':renewed,'model_leaf_changes':200,'fingerprint_mapping_counts':[{'before':a,'after':b,'count':n} for (a,b),n in counts.items()],'matrix_delta_count':len(md),'checks':['79 focused controls passed; lane source/artifact fingerprints unchanged.','Entire decoded original-to-candidate model has exactly200 fingerprint leaves matching persisted delta.','Complete fresh actual model semantically equals candidate under unchanged schema-addressed comparator.','Fresh observation and relation complete semantics equal committed artifacts.','All116 other payload bytes unchanged; matrix changes limited to source stamps and this model raw/logical hashes;39 summary rows unchanged.'],'preservation_runner_sha256':h(Path('/private/tmp/unfold-s81-approved-linux/verify_portable_proof_preservation.py').read_bytes()),'preservation_runner_disposition':'ACCEPT_SOURCE_ONLY; actual results pending','limits':['New owner approval required for exact two-file S7 artifact renewal; installation not authorized.','Local host evidence does not establish final Linux PASS; original failures retained.','Source correction is eight reviewed candidate files including six production files; proposal is separate two artifact files.']}
f=review/'actual-proposal-review.json';f.write_text(json.dumps(record,indent=2)+'\n');print(f,h(f.read_bytes()));print('mappingcounts',list(counts.values()),'matrixdelta',len(md))
