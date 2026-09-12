from pathlib import Path
import collections,difflib,gzip,hashlib,json,re,shutil,tarfile
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');B=R/'verification/receipts/S9-A-schema-candidate-fff99e22';S=Path('/private/tmp/unfold-s9a-thirtyfirst');D=S/'preservation-current31/recovered-deltas';O=B/'new-resume/reviews/current31-recovered-cause-audit';O.mkdir(exist_ok=False)
F=R/'.claude/worktrees/verify-s9-a-thirtyfirst';BASE=R/'.claude/worktrees/verify-s9-a-closure-baseline'
def sha(v):return hashlib.sha256(v).hexdigest()
def load(p):return json.loads(p.read_bytes())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def get(d,path):
 for k in path:d=d[int(k)] if isinstance(d,list) else d[k]
 return d
def parts(pointer):return [p.replace('~1','/').replace('~0','~') for p in pointer.split('/')[1:]]
def vh(value,status):return sha(json.dumps({'status':status,'value':value},sort_keys=True,separators=(',',':'),ensure_ascii=False,default=repr).encode())[:16]
def canon(v):return json.dumps(v,sort_keys=True,default=str).encode()
# Validate the independently archived recovered baseline before using its documents.
A=B/'thirtyfirst/preservation-baseline';mp=A/'archive-map.json';assert sha(mp.read_bytes())=='be2e560d5135d82679de6cebe135ca522e95dc2e1b30175095658a42dc2f8486';mapping=load(mp);index={v['path']:v for v in mapping['files']};assert sha((A/'exact-capture.tar.gz').read_bytes())==mapping['archive_sha256'];seen=set();raw_size=0
with tarfile.open(A/'exact-capture.tar.gz','r:gz') as tf:
 for member in tf:
  assert member.isfile() and member.name not in seen and member.name in index
  raw=tf.extractfile(member).read();row=index[member.name];assert sha(raw)==row['sha256'] and len(raw)==row['bytes'] and sha((S/'preservation-baseline'/member.name).read_bytes())==row['sha256'];seen.add(member.name);raw_size+=len(raw)
assert seen==set(index)
result=load(S/'preservation-baseline/result.json');assert result['preservation_passed'] and result['before']==result['after'] and result['artifacts_before']==result['artifacts_after']
recovery=load(D/'summary.json');assert recovery['all_surfaces_recovered'] and len(recovery['witnesses'])==29
reports={n:load(S/n/'capture/report.json') for n in ('preservation-baseline','preservation-current31')}
expected=load(S/'preservation-baseline/capture/controller/expected.executed.json');assert expected==load(S/'preservation-current31/capture/controller/expected.executed.json')
sources={'width':('model_unfolder/adapters/transformer/parser.py','model_hidden_size_value'), 'embedding':('model_unfolder/adapters/transformer/parser.py','_width_block'), 'mechanism':('model_unfolder/adapters/transformer/parser.py','attention_mechanism'), 'geometry':('model_unfolder/adapters/transformer/parser.py','attention_head_geometry'), 'activation':('model_unfolder/adapters/transformer/parser.py','_activation_event_status'), 'tie':('model_unfolder/adapters/transformer/parser.py','key="tie_word_embeddings"'), 'scope':('model_unfolder/evidence/registry.py','model_hidden_size_value'), 'scope_serialization':('model_unfolder/parser.py','receipted_scopes'), 'obligation':('model_unfolder/evidence/config_access.py','expected_value_status_hash'), 'references':('model_unfolder/evidence/context.py','presentation_reference'), 'reference_snapshot':('model_unfolder/presentation.py','def from_fact'), 'legacy_unknown':('model_unfolder/evidence/facts.py','legacy_unknown_reason_unrecorded'), 'native_unknown':('model_unfolder/evidence/facts.py','unknown_reason_unrecorded'), 'unknown':('model_unfolder/evidence/presentation_projection.py','def attach_unresolved_envelope_reasons'), 'finite_unknown_slots':('model_unfolder/presentation.py','def unresolved_spec_values'), 'chip':('model_unfolder/evidence/presentation_projection.py','def attach_fact_chips'), 'sable':('model_unfolder/sable.py','SableCheck("presentation_census"'), 'css':('model_unfolder/renderers/html/styles.py','Typed annotations remain distinguishable'), 'html_unknown':('model_unfolder/renderers/html/utils.py','def append_unknown_value_report'), 'banner':('model_unfolder/renderers/html/sections.py','stats_hidden_size'), 'html_chip':('model_unfolder/renderers/html/cards.py','presentation_chip_html'), 'payload':('model_unfolder/renderers/html/card_payload.py','digest = hashlib.sha256(fragment.encode())')}
source_records={}
for key,(path,needle) in sources.items():
 raw=(F/path).read_bytes();lines=raw.decode().splitlines();hits=[i+1 for i,line in enumerate(lines) if needle in line];assert hits,(key,path)
 source_records[key]={'path':path,'sha256':sha(raw),'matching_lines':hits,'needle':needle,'baseline_sha256':sha((BASE/path).read_bytes()) if (BASE/path).is_file() else None}
save(O/'source-causes.json',source_records)
all_counts=collections.Counter();all_findings=[];models={};html_counts=collections.Counter();hash_limits=0;hash_checks=0;css_packets=set();full_raw_counts=collections.Counter()
for slug,contract in sorted(recovery['witnesses'].items()):
 out=O/slug;out.mkdir();cases=[];docs=[];pages=[]
 for lane in reports:
  c=next(v for v in reports[lane]['cases'] if v['slug']==slug);cases.append(c)
  raw=(S/lane/'capture/controller'/c['actual_surfaces']['path']).read_bytes();assert sha(raw)==c['actual_surfaces']['sha256'];doc=json.loads(raw);docs.append(doc)
  for surface,h in expected['witnesses'][slug]['surfaces'].items():
   actual=sha(canon(doc[surface]));assert actual==c['surface_sha256'][surface]
   if lane=='preservation-baseline':assert actual==h
  page_row=next(v for v in c['renders'] if v['is_original_html_surface']);page_raw=(S/lane/'capture/controller'/page_row['actual']['path']).read_bytes();assert sha(page_raw)==page_row['actual']['sha256'];pages.append(page_raw.decode())
 before,after=docs;oldfacts,newfacts=[doc['ledgers']['fact_provenance'] for doc in docs]
 for key,row in oldfacts.items():
  for field in ('value','status','source'):assert row.get(field)==newfacts[key].get(field),(slug,key,field)
 assert set(newfacts)-set(oldfacts) in (set(),{'model.hidden_size'})
 scopesb,scopesa=[d['ledgers']['config_access']['projection_coverage']['receipted_scopes'] for d in docs];assert scopesa==sorted(scopesb+[['model','model_hidden_size_value']])
 rows=[];counts=collections.Counter();findings=[];verified_hash=0;unexposed_hash=0
 for surface in ('ir','ledgers'):
  changes=load(D/slug/(surface+'.json'));full_raw_counts[surface]+=len(changes)
  for number,change in enumerate(changes):
   p=parts(change['path']);value=change.get('after');cause=None;detail={};issues=[]
   if surface=='ir':
    assert change['operation']=='add'
    if p[-1]=='unknown_reason':
     cause='unknown';assert value=={'reason_class':'investigation_missing','concrete_reason':'unknown_reason_unrecorded','investigation':None};parent=get(before['ir'],p[:-1]);assert parent.get('resolved')is False or parent.get('kind')=='unknown'
    elif p==['extras','presentation_unresolved_values']:
     cause='finite_unknown_slots'
     for annotation in value:
      original=get(before['ir'],annotation['path']);assert original is None if annotation['value_state']=='null_unknown' else original=='unknown'
     detail={'existing_null_or_unknown_slots':len(value),'all_values_already_present_unchanged':True}
    elif p[-1]=='source_fact_keys':
     cause='embedding';assert value==['model.hidden_size'];parent=get(before['ir'],p[:-1]);assert parent['id']=='embed' and parent['kind']=='embedding';detail={'no_new_block':True,'citation':'model.hidden_size'}
    elif p[-1] in ('presentation_chips','presentation_path','presentation_aliases'):
     cause='chip';assert slug=='stable-diffusion-xl-base-1-0';block=get(after['ir'],p[:-1]);assert block['presentation_path']==[int(x) if str(x).isdigit() else x for x in p[:-1]] and block['presentation_aliases']==[]
     for chip in block['presentation_chips']:
      assert chip['chip_kind']=='class_default'
      for ref in chip['references']:
       assert ref==newfacts[ref['fact_key']]['presentation_reference'] and oldfacts[ref['fact_key']]['status']=='class_default';assert ref['fact_key'] in block['source_fact_keys']
     detail={'existing_default_fact_chip':True,'block_id':block['id']}
   elif p[:3]==['config_access','projection_coverage','receipted_scopes']:
    cause='scope';detail={'exact_sequence_is_sorted_baseline_plus_one':['model','model_hidden_size_value']}
   elif p[:2]==['config_access','projection_obligations']:
    ob=get(before['ledgers'],p[:3]);oa=get(after['ledgers'],p[:3]);cause={'attention_mechanism':'mechanism','attention_head_geometry':'geometry','ffn_activation':'activation','model_hidden_size_value':'width'}.get(oa['mechanism'])
    if oa['target']['key']=='tie_word_embeddings':cause='tie'
    detail={'before_obligation':ob,'after_obligation':oa}
    factkey=oa['target']['owner']+'.'+oa['target']['key'];fact=newfacts.get(factkey)
    if fact:
     ok=vh(fact['value'],fact['status'])==oa['expected_value_status_hash'];detail['actual_fact_value_status_hash_matches']=ok;detail['actual_fact_key']=factkey;verified_hash+=1
     if not ok:issues.append('event_hash_differs_from_actual_fact')
    else:unexposed_hash+=1;detail['limit']='Nested event remains saved but enclosing canonical surface has no child fact ledger; no child proof reconstructed.'
    if p[-2]=='source':
     oldsources=[x['source'] for x in before['ledgers']['config_access']['projection_obligations'] if x['mechanism']==oa['mechanism']];newsources=[x['source'] for x in after['ledgers']['config_access']['projection_obligations'] if x['mechanism']==oa['mechanism']]
     same=collections.Counter(map(lambda x:json.dumps(x,sort_keys=True),oldsources))==collections.Counter(map(lambda x:json.dumps(x,sort_keys=True),newsources));detail['source_multiset_equal_despite_order_change']=same
     if not same:issues.append('operand_source_multiset_changed')
   elif p[0]=='fact_provenance' and p[-1]=='presentation_reference':
    cause='references';fact=newfacts[p[1]];assert value['fact_key']==p[1] and value['status']==fact['status'] and value['value_status_hash']==vh(fact['value'],fact['status']);detail={'has_snapshot_proof':value['claim_proof']is not None,'source_role':'from_fact copies existing typed proof summary; serialization does not issue proofs'}
   elif p[0]=='fact_provenance' and p[-1]=='unknown_reason':
    cause='legacy_unknown' if value['concrete_reason']=='legacy_unknown_reason_unrecorded' else 'native_unknown';assert value['reason_class']=='investigation_missing' and value['investigation']is None
   elif p==['fact_provenance','model.hidden_size']:
    cause='width';assert value['value']==before['ir']['hidden_size']==after['ir']['hidden_size'];assert value['status']=='config_declared';proof=value['presentation_reference']['claim_proof'];assert proof['claim_kind']=='value' and proof['proof_kind']=='config_resolution';detail={'already_existing_scalar_value':value['value'],'claim_limit':'Exact checkpoint value; not embedding computation or connection proof.'}
   if cause is None:issues.append('unclassified_canonical_delta')
   row={'surface':surface,'raw_delta_index':number,'raw_path':change['path'],'cause':cause,'detail':detail,'findings':issues};rows.append(row);counts[cause or 'UNCLASSIFIED']+=1
   if issues:findings.append(row)
 # Exactly two inserted checks, all old outcomes/check bodies preserved.
 oldchecks,newchecks=[d['sable']['checks'] for d in docs];inserted=[x for x in newchecks if x['name'] in ('presentation_census','pending_design')];assert len(inserted)==2;assert [x for x in newchecks if x['name'] not in ('presentation_census','pending_design')]==oldchecks;assert before['sable']['mechanical_passed']==after['sable']['mechanical_passed'];assert all(x['passed'] and not x['findings'] for x in inserted)
 counts['sable']=len(load(D/slug/'sable.json'));full_raw_counts['sable']+=counts['sable'];assert counts['sable']==14
 # HTML canon uses exactly the established first-mount-only law; raw HTML diffs remain retained.
 canonical=[];mounts=[]
 for page in pages:
  mount=re.search(r'uf-[0-9a-f]{6,32}\b',page).group();mounts.append(mount);canonical.append(page.replace(mount,'<MOUNT>'))
 rawdiff=''.join(difflib.unified_diff(pages[0].splitlines(True),pages[1].splitlines(True),fromfile='baseline/raw.html',tofile='current31/raw.html'));(out/'html.raw.diff.gz').write_bytes(gzip.compress(rawdiff.encode(),mtime=0))
 aa,bb=[p.splitlines(True) for p in canonical];hunks=[];payload_delta=[];digests=[];payloads=[]
 pattern=r'<script type="application/json" data-uf-card-payload="([^"]+)" data-uf-card-mount="[^"]*">(.*?)</script>'
 for page in canonical:
  match=re.search(pattern,page)
  if match:digests.append(match.group(1));payloads.append(json.loads(match.group(2)))
 for tag,i,j,k,l in difflib.SequenceMatcher(a=aa,b=bb,autojunk=False).get_opcodes():
  if tag=='equal':continue
  left=''.join(aa[i:j]);right=''.join(bb[k:l]);cause=None
  if tag=='insert' and right.startswith('/* Typed annotations remain distinguishable'):
   cause='css';css_packets.add(right)
  elif left=='<div class="uf-card">\n' and right.startswith('<div class="uf-card"><details class="uf-unknown-values">') and right.endswith('</details>\n'):
   cause='html_unknown'
  elif left==right.replace(' data-uf-receipt-node="stats_hidden_size"','') and 'data-uf-receipt-node="stats_hidden_size"' in right:
   cause='banner'
  elif len(digests)==2 and left.replace(digests[0],digests[1])==right:
   cause='payload_digest_routing'
  elif len(payloads)==2 and left.startswith('<script type="application/json" data-uf-card-payload=') and right.startswith('<script type="application/json" data-uf-card-payload='):
   cause='payload'
  h={'tag':tag,'before_lines':[i+1,j],'after_lines':[k+1,l],'before':left,'after':right,'cause':cause};hunks.append(h);html_counts[cause or 'UNCLASSIFIED']+=1
  if cause is None:findings.append({'surface':'html','finding':'unclassified_HTML_hunk','before_lines':[i+1,j],'after_lines':[k+1,l]})
 if payloads:
  assert slug=='stable-diffusion-xl-base-1-0';pb,pa=payloads;assert set(pb)==set(pa)
  for key in pb:
   if key not in ('templates','canonical'):assert pb[key]==pa[key],key
  assert len(pb['templates'])==len(pa['templates']) and len(pb['canonical'])==len(pa['canonical'])
  template_changes=[i for i,(x,y) in enumerate(zip(pb['templates'],pa['templates'])) if x!=y];canonical_changes=[i for i,(x,y) in enumerate(zip(pb['canonical'],pa['canonical'])) if x!=y];assert template_changes==[7] and canonical_changes==[0]
  assert len(pb['templates'][7])==len(pa['templates'][7]) and pb['templates'][7][1:]==pa['templates'][7][1:]
  assert 'uf-chip-class_default' in pa['templates'][7][0] and 'uf-chip-class_default' not in pb['templates'][7][0]
  payload_delta=[{'path':'/templates/7/0','before':pb['templates'][7][0],'after':pa['templates'][7][0],'cause':'html_chip'}, {'path':'/canonical/0','before':pb['canonical'][0],'after':pa['canonical'][0],'cause':'css_and_html_unknown_in_canonical_payload'}]
  save(out/'payload-deltas.json',payload_delta)
 save(out/'canonical-delta-causes.json',rows);save(out/'html-hunks.json',hunks)
 record={'canonical_delta_counts':{s:contract['surfaces'][s]['delta_count'] for s in contract['surfaces']},'causes':dict(counts),'all_existing_fact_value_status_source_triples_unchanged':True,'new_fact_keys':sorted(set(newfacts)-set(oldfacts)),'new_sable_checks':inserted,'matched_actual_fact_event_hash_rows':verified_hash,'unexposed_nested_event_hash_rows':unexposed_hash,'html_hunk_counts':dict(collections.Counter(v['cause'] or 'UNCLASSIFIED' for v in hunks)),'original_mounts':mounts,'raw_html_sha256':[sha(p.encode()) for p in pages],'unknown_report_counts':[int(m.group(1)) for page in canonical for m in re.finditer(r'Unresolved values · (\d+) ·',page)],'findings':findings}
 save(out/'summary.json',record);models[slug]=record;all_counts.update(counts);hash_checks+=verified_hash;hash_limits+=unexposed_hash;all_findings.extend({'slug':slug,**f} for f in findings)
assert len(css_packets)==1
summary={'verdict':'EXACT_RECOVERED_CAUSE_AUDIT_NOT_ACCEPTANCE','recovered_baseline':{'map_sha256':sha(mp.read_bytes()),'archive_members_verified':len(seen),'raw_bytes':raw_size,'all203_surfaces_match_committed_expected':True,'original_test_pass_count':52,'duration_seconds':result['duration']},'canonical_delta_totals':dict(full_raw_counts),'source_cause_totals':dict(all_counts),'html_hunk_counts':dict(html_counts),'html_hash_changes':29,'existing_fact_triples_unchanged_all29':True,'new_model_hidden_size_scalar_facts':sum('model.hidden_size'in v['new_fact_keys'] for v in models.values()),'matched_actual_fact_event_hash_rows':hash_checks,'unexposed_nested_event_hash_rows':hash_limits,'models':models,'findings':all_findings,'scope_limits':['Full raw canonical deltas and HTML hunks retained. Hash matching baseline establishes comparison identity, not approval of changes.', 'Nested event hashes cannot be checked against a child native ledger absent from enclosing canonical documents; no proof is reconstructed.', 'Presentation-reference snapshots do not upgrade native facts; added model.hidden_size facts establish only scalar config values.', 'HTML source causes are finite insertion/attribute/payload classifications; browser visibility, click behavior and layout acceptance remain independent.', 'SDXL payload arrays retain identical svgs/identities/cards/containers; two string leaves change (class-default-chip template and canonical shell annotations). No model is rerun by this audit.', 'Current31 predates the separately identified synthetic false-softcap-receipt fix; unchanged29 witness surfaces do not erase that bounded finding.'], 'output_approval':False,'runtime':'Stdlib saved-data/source reads only.'}
save(O/'summary.json',summary);(O/'audit-source.py').write_bytes(Path('/private/tmp/audit-recovered31-causes.py').read_bytes())
# Preserve every original comparator delta without modifying either completed packet.
with tarfile.open(O/'exact-comparator-deltas.tar.gz','w:gz') as tf:
 for p in sorted(D.rglob('*')):
  if p.is_file():tf.add(p,arcname=p.relative_to(D),recursive=False)
save(O/'manifest.json',{'files':[{'path':str(p.relative_to(O)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(O.rglob('*')) if p.is_file()]})
print(json.dumps({'manifest_sha256':sha((O/'manifest.json').read_bytes()),'summary_sha256':sha((O/'summary.json').read_bytes()),'causes':dict(all_counts),'html':dict(html_counts),'hash_checks':hash_checks,'hash_limits':hash_limits,'findings':len(all_findings)},indent=2))
