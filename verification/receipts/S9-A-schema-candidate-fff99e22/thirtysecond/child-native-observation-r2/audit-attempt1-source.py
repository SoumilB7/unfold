from pathlib import Path
import collections,gzip,hashlib,json,tarfile
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');B=R/'verification/receipts/S9-A-schema-candidate-fff99e22';RAW=Path('/private/tmp/unfold-s9a-thirtysecond/child-native-observation-r2');O=B/'thirtysecond/child-native-observation-r2';O.mkdir(exist_ok=False);H=B/'new-resume/child-native-observation32-proposal-r2';PRIOR=Path('/private/tmp/unfold-s9a-thirtysecond/current-pages/capture')
def sha(data):return hashlib.sha256(data).hexdigest()
def load(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def value_hash(v,s):return sha(json.dumps({'status':s,'value':v},sort_keys=True,separators=(',',':'),ensure_ascii=False,default=repr).encode())[:16]
lane=load(RAW/'result.json');assert lane['passed']and lane['packet_passed']and lane['pins_equal']and lane['before']==lane['after']and lane['artifacts_before']==lane['artifacts_after'];pins=load(RAW/'pins-before.json');assert pins==load(RAW/'pins-after.json');assert pins['proposal']['manifest.json']==sha((H/'manifest.json').read_bytes());held=load(H/'manifest.json');assert all(sha((H/name).read_bytes())==digest for name,digest in held['files'].items())
files={p.relative_to(RAW).as_posix():{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}for p in sorted(RAW.rglob('*'))if p.is_file()}
with tarfile.open(O/'exact-capture.tar.gz','w:gz')as archive:
 for name in files:archive.add(RAW/name,arcname=name,recursive=False)
with tarfile.open(O/'exact-capture.tar.gz','r:gz')as archive:
 assert {m.name for m in archive.getmembers()}==set(files)
 for m in archive.getmembers():
  data=archive.extractfile(m).read();assert sha(data)==files[m.name]['sha256']and len(data)==files[m.name]['bytes']
packet=load(RAW/'capture/result.json');assert packet['passed']and packet['pins_equal']and packet['totals']=={'models':15,'obligations':105,'targets':80}
wanted={(r['slug'],r['index']):r for r in load(H/'obligations.json')};alljoins={};targets={};byfact=collections.defaultdict(collections.Counter);modelrows=[];tuple_deltas=[];root_type_deltas=[];live_snapshot_pairs=0;allnativeproofs=collections.Counter();targetproofs=collections.Counter();indexrels=collections.Counter();declarationcount=0
for model in packet['models']:
 slug=model['slug'];dest=RAW/'capture'/slug;assert model['passed']and not model['errors']and model['product_error']is None and model['root_parse_calls']==1
 for name,digest in model['artifact_sha256'].items():assert sha((dest/name).read_bytes())==digest
 assert all(c['original_calls']==1 and c['original_returned']and c['return_identity_preserved']for c in model['projector_calls'])
 root_ir=load(dest/'root-ir.json');previous_ir=json.loads(gzip.decompress((PRIOR/slug/'ir-before-render.json.gz').read_bytes()));assert root_ir==previous_ir and load(dest/'root-saved-json-deltas.json')==[]
 root_diff=load(dest/'root-live-vs-saved-json-deltas.json');assert all(d['kind']=='type_changed'and d['before_type']=='tuple'and d['after_type']=='list'and d['before']==d['after']for d in root_diff);root_type_deltas.extend({'slug':slug,**d}for d in root_diff)
 root_input=load(PRIOR/slug/'input.json');children=load(dest/'child-ledgers-local.json');assert len(children)==len(model['projector_calls'])==model['captured_children'];seen_model=[]
 for child in children:
  ns=child['namespace'];binding=child['binding'];assert binding['owner']==ns and binding['exact_live_binding_identity_verified'];raw_child=root_input
  for key in binding['document_path']:raw_child=raw_child[key]
  assert raw_child==binding['checkpoint']and binding['document']==binding['checkpoint']
  snap=dest/child['live_snapshot_directory'];before=load(snap/'before.json');after=load(snap/'after.json');assert before==after and load(snap/'live-before-after-deltas.json')==[]and child['live_snapshot_delta_count']==0;live_snapshot_pairs+=1
  ds=load(snap/'json-serialization-type-deltas.json');assert all(d['kind']=='type_changed'and d['before_type']=='tuple'and d['after_type']=='list'and d['before']==d['after']for d in ds)
  type_paths={r['path']:r['python_type']for r in before['python_types']if 'python_type'in r}
  for d in ds:assert type_paths[d['path']]=='tuple'
  tuple_deltas.extend({'slug':slug,'namespace':ns,**d}for d in ds)
  assert before['json_value']['records']==child['records']
  for key,native in child['native_declarations'].items():
   declarationcount+=1;proof=native['claim_proof'];state=native['existing_proof_validation'];allnativeproofs[state]+=1;assert state==('MISSING_UNCHANGED'if proof is None else'VALIDATED')
   assert native['qualified_key']==ns+'.'+key and key==native['owner']+'.'+native['key'];rec=child['records'][key];assert native['value']==rec['value']and native['status']==rec['status'];assert value_hash(native['value'],native['status'])==native['value_status_hash'];assert rec['presentation_reference']['claim_proof']==proof
   if proof is not None:assert proof['fact_id']==key and proof['claim_kind']==native['claim_kind']and proof['reader_symbols']==native['claim_readers']
  for join in child['joins']:
   identity=(slug,join['index']);assert identity in wanted and identity not in alljoins;w=wanted[identity];ob=w['obligation'];key=join['local_native_key'];native=child['native_declarations'][key];event=join['event'];qualified=join['qualified_key'];assert qualified==w['qualified_key']==ns+'.'+key==ob['target']['owner']+'.'+ob['target']['key'];assert ob['source']['component']==ns
   assert root_ir['extras']['config_access']['projection_obligations'][join['index']]==ob
   assert event['component']==ns and event['document_path']==binding['document_path']and event['intent']=='consumed'
   assert event['config_path']==ob['source']['path']and event['canonical']==ob['source']['canonical']and(event['alias']or event['canonical'])==ob['source']['spelling']
   assert event['fact_owner']==ob['target']['owner']and event['fact_key']==ob['target']['key']and event['mechanism']==ob['mechanism']
   assert event['value_status_hash']==value_hash(native['value'],native['status'])==ob['expected_value_status_hash']==join['native_value_status_hash']
   assert child['active_ledger_events_local'].count(event)==1 and len(event['document_fingerprint'])==64 and len(event['document_token'])==64
   assert join['proof_validation']==native['existing_proof_validation'];targets[(slug,qualified)]=native;targetproofs[join['proof_validation']]+=1;byfact[native['key']][join['proof_validation']]+=1;indexrels[native['proof_index_relation_to_actual_child_context']]+=1
   alljoins[identity]={'slug':slug,'index':join['index'],'qualified_key':qualified,'namespace':ns,'native_value':native['value'],'native_status':native['status'],'native_hash':native['value_status_hash'],'source':ob['source'],'target':ob['target'],'mechanism':ob['mechanism'],'document_path':binding['document_path'],'existing_proof_validation':join['proof_validation'],'claim_kind':native['claim_kind'],'event':event};seen_model.append(identity)
  assert child['joins']==[r for r in load(dest/'joins.json')if r['event']['component']==ns]
 assert len(seen_model)==model['joined_obligations'];assert len({alljoins[k]['qualified_key']for k in seen_model})==model['joined_targets'];modelrows.append({k:model[k]for k in ('slug','captured_children','joined_obligations','joined_targets','existing_proof_validation','serialized_root_ir_equal_saved_actual32')})
assert set(alljoins)==set(wanted)and len(alljoins)==105 and len(targets)==80 and live_snapshot_pairs==20
original147=load(B/'new-resume/reviews/nested147-current32-audit/all147-rows.json');closed=[]
for row in original147:
 identity=(row['slug'],int(row['raw_canonical_delta_path'].split('/')[3]));join=alljoins[identity];assert row['qualified_child_fact_key']==join['qualified_key'];closed.append({'slug':row['slug'],'raw_canonical_delta_path':row['raw_canonical_delta_path'],'cause':row['cause'],'field_changed':row['field_changed'],'native_hash_limit':'CLOSED_BY_ACTUAL_BOUND_CHILD_NATIVE_RECORD','join':join})
assert len(closed)==147
summary={'verdict':'PASS_BOUND_CHILD_NATIVE_HASH_LIMIT_CLOSED_147_ROWS','lane':lane,'original_canonical_delta_rows':147,'unique_obligations':105,'unique_native_targets':80,'models':15,'actual_child_contexts':20,'all_original_calls_once_and_return_identity_preserved':True,'live_before_after_snapshot_pairs_exact_values_and_types':20,'live_mutation_delta_count':0,'root_saved_json_delta_count':0,'child_json_serialization_tuple_to_list_diagnostics':len(tuple_deltas),'root_live_vs_saved_json_tuple_to_list_diagnostics':len(root_type_deltas),'target_proofs_by_obligation':dict(targetproofs),'target_proofs_by_unique_target':dict(collections.Counter(d['existing_proof_validation']for d in targets.values())),'by_fact_key_obligation':{k:dict(v)for k,v in byfact.items()},'all_captured_native_declarations':declarationcount,'all_captured_native_existing_proof_states':dict(allnativeproofs),'target_index_relation_by_obligation':dict(indexrels),'proof_scope':'30 unique mechanism/geometry target facts (55 obligations) were validated live from existing proofs. 50 unique target facts (50 obligations) retain no proof:20 hidden_size,17 activation,13 tie_word_embeddings. The saved audit verifies exact data joins and unchanged snapshots; it does not replay process-local proof identity or upgrade missing proofs.','accuracy_limit':'This closes only the native value/status hash and exact association validation limit for147 recorded delta rows. It does not prove missing mechanisms, drawing presence, browser visibility, or S9-A completion. It does not rewrite original RED R1 or the original unverified audit.','R1_failure_cause_now_observed_in_R2':'Live records include tuple-valued ffn_schedule/mask_schedule; returned sub_model schedule runs also contain tuples. Full exact paths are retained in diagnostic inventory. R1 did not persist its own child local types; R2 supplies current observations, never retroactively relabels R1 PASS.','runtime_by_root_only':True,'saved_data_audit_performed_no_product_imports_or_tests':True,'output_approval':False,'archive':{'files':len(files),'raw_bytes':sum(r['bytes']for r in files.values()),'all_members_rehashed':True,'tar_sha256':sha((O/'exact-capture.tar.gz').read_bytes())},'per_model':modelrows}
save(O/'artifact-map.json',files);save(O/'summary.json',summary);save(O/'all105-native-joins.json',list(alljoins.values()));save(O/'all147-hash-limit-closures.json',closed);save(O/'child-serialization-type-deltas.json',tuple_deltas);save(O/'root-live-vs-json-type-deltas.json',root_type_deltas);(O/'audit-source.py').write_bytes(Path('/private/tmp/audit-child-observer32-r2.py').read_bytes());save(O/'manifest.json',{'files':{p.name:sha(p.read_bytes())for p in sorted(O.iterdir())if p.is_file()}});print(json.dumps({'manifest_sha256':sha((O/'manifest.json').read_bytes()),'archive':summary['archive'],'proofs':summary['target_proofs_by_unique_target'],'live_pairs':live_snapshot_pairs,'serialization_types':len(tuple_deltas),'root_types':len(root_type_deltas)},indent=2))
