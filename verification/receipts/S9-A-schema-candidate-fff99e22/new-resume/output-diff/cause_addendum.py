"""Saved-output/source cause predicates. No product imports or acceptance normalization."""
from pathlib import Path
import collections, difflib, gzip, hashlib, json, re
ROOT=Path(__file__).resolve().parents[5]
# Locate the package without importing it.
while not (ROOT/'model_unfolder').is_dir(): ROOT=ROOT.parent
FROZEN=ROOT/'.claude/worktrees/verify-s9-a-twentysixth'
BASE=ROOT/'.claude/worktrees/verify-s9-a-closure-baseline'
SCRATCH=Path('/private/tmp/unfold-s9a-twentysixth')
OUT=Path(__file__).parent/'cause-addendum'
OUT.mkdir(exist_ok=False)
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def write(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def get(d,p):
 for k in p:d=d[k]
 return d

def vh(value,status): return sha(json.dumps({'status':status,'value':value},sort_keys=True,separators=(',',':'),ensure_ascii=False,default=repr).encode())[:16]
source_ranges={
 'width':('model_unfolder/adapters/transformer/parser.py',1422,1450),
 'embedding':('model_unfolder/adapters/transformer/parser.py',4823,4832),
 'mechanism':('model_unfolder/adapters/transformer/parser.py',870,944),
 'geometry':('model_unfolder/adapters/transformer/parser.py',2710,2880),
 'activation':('model_unfolder/adapters/transformer/parser.py',1590,1725),
 'tie':('model_unfolder/adapters/transformer/parser.py',4500,4540),
 'scope':('model_unfolder/evidence/registry.py',251,268),
 'scope_serialization':('model_unfolder/parser.py',367,380),
 'obligation':('model_unfolder/evidence/config_access.py',631,666),
 'references':('model_unfolder/evidence/context.py',142,157),
 'reference_snapshot':('model_unfolder/presentation.py',43,82),
 'legacy_unknown':('model_unfolder/evidence/facts.py',331,357),
 'native_unknown':('model_unfolder/evidence/facts.py',187,201),
 'unknown':('model_unfolder/evidence/presentation_projection.py',10,31),
 'chip':('model_unfolder/evidence/presentation_projection.py',35,113),
 'activation_source':('model_unfolder/evidence/sources.py',251,292),
 'supporting_source_callers':('model_unfolder/evidence/sources.py',135,229),
 'index_supporting_sources':('model_unfolder/evidence/program_index.py',3268,3336),
}
sources={}
for key,(path,start,end) in source_ranges.items():
 b=(FROZEN/path).read_bytes();lines=b.decode().splitlines();old=(BASE/path).read_bytes() if (BASE/path).exists() else b''
 sources[key]={'path':path,'sha256':sha(b),'lines':[start,end],'excerpt':'\n'.join(lines[start-1:end]),'baseline_sha256':sha(old) if old else None}
 dest=OUT/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 patch=OUT/'source-diffs'/(path+'.patch');patch.parent.mkdir(parents=True,exist_ok=True);patch.write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),b.decode().splitlines(True),fromfile='fff99e22/'+path,tofile='frozen26/'+path)))
write(OUT/'sources.json',sources)
allmodels=[];counts=collections.Counter();unexplained=[];closure_details=[]
for deltafile in sorted((SCRATCH/'output-delta-enumeration').rglob('ir-before-render-deltas.json.gz')):
 slug=deltafile.parent.name;bdir=SCRATCH/'source-closure-replay/before/capture'/slug;adir=SCRATCH/'current-pages/capture'/slug
 before=read(bdir/'ir-before-render.json.gz');after=read(adir/'ir-before-render.json.gz');facts=read(adir/'facts.json');deltas=read(deltafile)
 assert deltas==read(deltafile.with_name('ir-after-render-deltas.json.gz'))
 fp=after['extras']['fact_provenance'];modelcounts=collections.Counter();rows=[]
 scopesb=before['extras']['config_access']['projection_coverage']['receipted_scopes'];scopesa=after['extras']['config_access']['projection_coverage']['receipted_scopes']
 assert scopesa==sorted(scopesb+[['model','model_hidden_size_value']])
 for number,d in enumerate(deltas):
  p=d['path'];v=d['after']; cause=None; detail={};validation=[]
  if p[:4]==['extras','config_access','projection_coverage','receipted_scopes']:
   cause='scope';detail={'inserted_scope':['model','model_hidden_size_value'],'complete_sequence_equals_sorted_baseline_plus_one':True}
  elif p[:3]==['extras','config_access','projection_obligations']:
   ob=get(before,p[:4]);oa=get(after,p[:4]); mech=oa['mechanism']; key=oa['target']['key']
   cause={'attention_mechanism':'mechanism','attention_head_geometry':'geometry','ffn_activation':'activation','model_hidden_size_value':'width'}.get(mech)
   if key=='tie_word_embeddings':cause='tie'
   detail={'before_obligation':ob,'after_obligation':oa,'serializer_source':'obligation'}
   # Hash-check against actual fact when that exact fact is exposed in this enclosing IR.
   factkey=oa['target']['owner']+'.'+key;f=fp.get(factkey)
   if f:
    expected=vh(f['value'],f['status']);detail['actual_fact_key']=factkey;detail['expected_hash_matches_actual_fact']=expected==oa['expected_value_status_hash']
    if expected!=oa['expected_value_status_hash']:validation.append('event_hash_disagrees_with_actual_fact')
   else:detail['actual_fact_limit']='Nested parser event exists but enclosing IR exposes only its own fact ledger; no fabricated child proof.'
   if p[-2:-1]==['source']:
    old_group=[x['source'] for x in before['extras']['config_access']['projection_obligations'] if x['mechanism']==mech]
    new_group=[x['source'] for x in after['extras']['config_access']['projection_obligations'] if x['mechanism']==mech]
    detail['source_multiset_preserved']=collections.Counter(map(lambda x:json.dumps(x,sort_keys=True),old_group))==collections.Counter(map(lambda x:json.dumps(x,sort_keys=True),new_group))
  elif p[:2]==['extras','fact_provenance'] and p[-1]=='presentation_reference':
   cause='references';f=fp[p[2]];assert not d['before_present'];assert v==facts[p[2]]['presentation_reference'];assert v['fact_key']==p[2];assert v['status']==f['status'];assert v['value_status_hash']==vh(f['value'],f['status']);detail={'reference_snapshot_source':'reference_snapshot','proof_kind':v['claim_proof']['proof_kind'] if v['claim_proof'] else None,'claim_kind':v['claim_proof']['claim_kind'] if v['claim_proof'] else None,'proof_was_not_invented_by_serializer':True}
  elif p[:2]==['extras','fact_provenance'] and p[-1]=='unknown_reason':
   cause='legacy_unknown' if v['concrete_reason']=='legacy_unknown_reason_unrecorded' else 'native_unknown';assert not d['before_present'];assert v['reason_class']=='investigation_missing' and v['investigation'] is None and v['concrete_reason'] in {'legacy_unknown_reason_unrecorded','unknown_reason_unrecorded'}
  elif p==['extras','fact_provenance','model.hidden_size']:
   cause='width';assert not d['before_present'];assert v['status']=='config_declared';assert v['value']==before['hidden_size']==after['hidden_size'];proof=v['presentation_reference']['claim_proof'];assert proof['claim_kind']=='value' and proof['proof_kind']=='config_resolution';detail={'existing_model_hidden_size':before['hidden_size'],'new_fact':v,'limit':'Exact scalar declaration only; no embedding execution or state-flow connection proof.'}
  elif p[:3]==['extras','render','model_blocks'] and p[-1]=='source_fact_keys':
   cause='embedding';assert not d['before_present'];assert v==['model.hidden_size'];block=get(before,p[:-1]);assert block['id']=='embed' and block['kind']=='embedding';assert fp['model.hidden_size']['presentation_reference']['claim_proof'];detail={'existing_block_id':block['id'],'existing_kind':block['kind'],'new_reference':'model.hidden_size','no_block_inserted':True}
  elif p[-1]=='unknown_reason':
   cause='unknown';assert not d['before_present'];assert v=={'reason_class':'investigation_missing','concrete_reason':'unknown_reason_unrecorded','investigation':None};parent=get(before,p[:-1]);assert parent.get('resolved') is False or parent.get('kind')=='unknown';detail={'before_resolved':parent.get('resolved'),'before_kind':parent.get('kind')}
  elif p==['extras','presentation_unresolved_values']:
   cause='unknown';assert not d['before_present']
   for item in v:
    old=get(before,item['path']);assert old is None if item['value_state']=='null_unknown' else old=='unknown'
   detail={'existing_unresolved_slots':len(v),'all_values_already_null_or_literal_unknown':True,'records':v}
  elif p[-1] in {'presentation_chips','presentation_path','presentation_aliases'}:
   cause='chip';assert not d['before_present'];block=get(after,p[:-1]);assert block['presentation_path']==p[:-1];assert block['presentation_aliases']==[]
   for chip in block['presentation_chips']:
    assert chip['chip_kind']=='class_default'
    for ref in chip['references']:
     assert ref==fp[ref['fact_key']]['presentation_reference'];assert ref['fact_key'] in block['source_fact_keys'];assert before['extras']['fact_provenance'][ref['fact_key']]['value']==fp[ref['fact_key']]['value'];assert before['extras']['fact_provenance'][ref['fact_key']]['status']=='class_default'
   detail={'block_id':block.get('id'),'chips':block['presentation_chips'],'previously_existing_default_fact':True}
  if cause is None: validation.append('no_finite_source_cause_identified')
  status='CAUSE_ATTRIBUTED_NOT_ACCEPTED' if not validation else 'REQUIRES_REVIEW'
  row={'raw_delta_index':number,'path':p,'cause':cause,'status':status,'details':detail,'findings':validation};rows.append(row);modelcounts[cause or 'UNEXPLAINED']+=1
  if validation:unexplained.append({'slug':slug,**row})
 # Existing fact triples must remain unchanged; added scalar fact is explicitly enumerated above.
 oldfp=before['extras']['fact_provenance'];assert set(oldfp)<=set(fp)
 for key,f in oldfp.items():
  for field in ['value','status','source']:assert f.get(field)==fp[key].get(field)
 record={'slug':slug,'raw_deltas':len(deltas),'counts':dict(modelcounts),'existing_fact_triples_unchanged':True,'new_fact_keys':sorted(set(fp)-set(oldfp)),'rows':rows}
 write(OUT/(slug+'.json'),record);counts.update(modelcounts);allmodels.append({k:v for k,v in record.items() if k!='rows'})
 # Follow recorded supporting source addresses, then verify installed bytes match exact portable additions.
 bd=read(bdir/'source-bundle-after-local.json')['bundle']; cd=SCRATCH/'source-closure-replay/after/capture'/slug;ad=read(cd/'source-bundle-after-local.json')['bundle'];closure=read(deltafile.parent/'closure-deltas.json');crows=[]
 for addition in closure['added']:
  if addition['content_sha256']!='4b1a469c24be4e4c6320536ad55a7337d24de04c3ecc71597fbf0267f1560612':continue
  component=addition['component']; paths=ad['supporting_files'].get(component,[]);candidates=[Path(p) for p in paths if Path(p).name=='activations.py' and sha(Path(p).read_bytes())==addition['content_sha256']];assert len(candidates)==1;activation=candidates[0];assert str(activation) not in bd.get('supporting_files',{}).get(component,[])
  matching=[]
  for modelingpath in ad['component_files'].get(component,[]):
   modeling=Path(modelingpath)
   if modeling.parent.parent.parent/'activations.py' != activation:continue
   for familyfile in sorted(modeling.parent.glob('modeling*.py')):
    source=familyfile.read_text();match=re.search(r'^from\s+(?:\.{2,}activations|transformers\.activations)\s+import\s+.*\b(?:ACT2FN|get_activation)\b',source,re.M)
    if match:
     matching.append({'modeling_locator':str(familyfile.relative_to(activation.parent)),'modeling_sha256':sha(familyfile.read_bytes()),'line':source[:match.start()].count('\n')+1,'import_text':match.group(0)});break
  assert matching
  crows.append({'added_portable_row':addition,'supporting_file_locator':'transformers/activations.py','supporting_file_sha256':sha(activation.read_bytes()),'recorded_component':component,'source_rule':'activation_source','matching_family_imports':matching,'limit':'Source-address closure addition, not a mechanism claim or evidence that selected activation executes.'})
 closure_details.append({'slug':slug,'raw_added_rows':closure['added'],'raw_removed_rows':closure['removed'],'activation_addition_causes':crows,'before_seal':closure['before_seal'],'after_seal':closure['after_seal']})
write(OUT/'closure-causes.json',closure_details)
write(OUT/'summary.json',{'verdict':'CAUSE_AUDIT_NOT_OUTPUT_ACCEPTANCE','models':allmodels,'counts':dict(counts),'raw_ir_deltas_per_phase':sum(counts.values()),'remaining_findings':unexplained,'activation_content_additions_explained':sum(len(r['activation_addition_causes']) for r in closure_details),'limits':['Source cause attribution does not independently execute or validate proof object internals; original actual parser validation and reviewer verdict remain necessary.','Nested event facts not published in enclosing IR retain explicit fact-hash verification limits.','Full HTML deltas beyond finite SVG identity audit are not covered by this IR/source report.','Original delta paths/bytes are retained unchanged; supplementary scope/source multiset comparisons explain ordering, never normalize acceptance.']})
print(json.dumps({'counts':counts,'unexplained':len(unexplained),'activation_additions':sum(len(r['activation_addition_causes']) for r in closure_details)},indent=2))
