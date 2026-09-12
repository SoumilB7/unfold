from pathlib import Path
import collections,difflib,gzip,hashlib,json,re
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');B=R/'verification/receipts/S9-A-schema-candidate-fff99e22';A=B/'thirtyfourth/current-pages';RAW=Path('/private/tmp/unfold-s9a-thirtyfourth/current-pages');OLD=Path('/private/tmp/unfold-s9a-thirtysecond/current-pages/capture');H=B/'new-resume/reviews/current34-saved-audit';O=B/'new-resume/reviews/current34-saved-independent-event';O.mkdir(exist_ok=False);F=R/'.claude/worktrees/verify-s9-a-thirtyfourth'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(p):return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def vh(value,status):return sha(json.dumps({'status':status,'value':value},sort_keys=True,separators=(',',':'),ensure_ascii=False,default=repr).encode())[:16]
def diff(a,b,p=()):
 if type(a)is type(b)and a==b:return[]
 if isinstance(a,dict)and isinstance(b,dict):
  rows=[]
  for k in sorted(a.keys()|b.keys()):
   if k in a and k in b:rows+=diff(a[k],b[k],(*p,k))
   else:rows.append({'path':list((*p,k)),'before_present':k in a,'after_present':k in b,'before':a.get(k),'after':b.get(k)})
  return rows
 if isinstance(a,list)and isinstance(b,list):
  rows=[]
  for i in range(max(len(a),len(b))):
   if i<len(a)and i<len(b):rows+=diff(a[i],b[i],(*p,i))
   else:rows.append({'path':list((*p,i)),'before_present':i<len(a),'after_present':i<len(b),'before':a[i]if i<len(a)else None,'after':b[i]if i<len(b)else None})
  return rows
 return[{'path':list(p),'before_present':True,'after_present':True,'before':a,'after':b,'before_type':type(a).__name__,'after_type':type(b).__name__}]
assert sha((A/'archive-map.json').read_bytes())=='1bf95f4cbf0ac82e113b6fbeb108998742658ee19d6004d1cff4d487d68fa028';archive=load(A/'archive-map.json');assert len(archive)==590
for row in archive:
 p=A/row['path'];q=RAW/row['path'];assert len(p.read_bytes())==row['bytes']and sha(p.read_bytes())==row['sha256']and p.read_bytes()==q.read_bytes()
assert load(A/'pins-before.json')==load(A/'pins-after.json');lane=load(A/'result.json');assert lane['passed']and lane['pins_equal']and lane['before']==lane['after']=='b9a2c4d8937c1d53b65baac166e9cc0be2a86d7a5c63c2cd25c637ce2dd3b75e'and lane['artifacts_before']==lane['artifacts_after']=='5aed5480485a877c5393ec4eb97be56cf2709327baa35ede0da54884029bd97e'
assert sha((H/'manifest.json').read_bytes())=='bdb3e26257adef6de7f2631794e2295f336051180822c834e3f4ba4b584774f0';assert all(sha((H/r['path']).read_bytes())==r['sha256']for r in load(H/'manifest.json')['files'])
fleet=load(A/'capture/result.json');assert fleet['captured_count']==39 and fleet['failed_count']==0 and len(fleet['models'])==39;audit=load(H/'results/summary.json');assert len(audit['models'])==39
for k in ('metadata_findings','missing_rendered_paths','unexpected_rendered_paths','class3_records','capture_census_findings'):assert audit['totals'][k]==0
native_changes=[];phase_totals=collections.Counter();htmlrows=[];special={};allmodels=[];rootcount=0
for model in fleet['models']:
 slug=model['slug'];new=A/'capture'/slug;old=OLD/slug;dest=O/slug;dest.mkdir()
 for name,digest in model['artifact_sha256'].items():assert sha((new/name).read_bytes())==digest
 a=load(old/'declarations.json');z=load(new/'declarations.json');delta=diff(a,z);save(dest/'native-deltas.json',delta)
 changed=sorted(k for k in a.keys()|z.keys()if a.get(k)!=z.get(k));native_changes.extend((slug,k)for k in changed)
 phase_deltas={}
 for name in ('ir-before-render.json.gz','ir-after-render.json.gz','diagram-ir-before-render.json.gz','diagram-ir-after-render.json.gz'):
  before=load(old/name);after=load(new/name);d=diff(before,after);phase_deltas[name]=d;phase_totals[name]+=len(d)
  assert before['layers']==after['layers']
  assert all(r['path'][:2]in [['extras','fact_provenance'],['extras','config_access']]for r in d)
 save(dest/'all-phase-ir-deltas.json',phase_deltas)
 rowaudit=load(H/'results'/f'{slug}.json');assert {r['path']:r['sha256']for r in rowaudit['inputs']}=={p.name:sha(p.read_bytes())for p in sorted(new.iterdir())if p.is_file()}
 pages=[gzip.decompress((p/'page.html.gz').read_bytes()).decode()for p in (old,new)];ids=[re.findall(r'<div id="(uf-[0-9a-f]{10})" class="uf-root">',page)for page in pages];assert all(len(v)==1 for v in ids);beforeid,afterid=ids[0][0],ids[1][0]
 substituted=pages[0].replace(beforeid,afterid);assert substituted==pages[1] # finite cause check only; original bytes/hashes/diff retained
 rawdiff=''.join(difflib.unified_diff(pages[0].splitlines(True),pages[1].splitlines(True),fromfile='actual32/'+slug+'.html',tofile='actual34/'+slug+'.html'))
 (dest/'raw-html.diff.gz').write_bytes(gzip.compress(rawdiff.encode(),mtime=0));svgs=[re.findall(r'<svg\b.*?</svg>',page,re.S)for page in pages];assert [s.replace(beforeid,afterid)for s in svgs[0]]==svgs[1]
 payloads=[re.findall(r'<script\b[^>]*data-uf-card-payload[^>]*>(.*?)</script>',page,re.S)for page in pages];assert len(payloads[0])==len(payloads[1]);payload_deltas=[diff(json.loads(x),json.loads(y))for x,y in zip(*payloads)]
 htmlrow={'slug':slug,'before_sha256':sha(pages[0].encode()),'after_sha256':sha(pages[1].encode()),'before_bytes':len(pages[0].encode()),'after_bytes':len(pages[1].encode()),'raw_equal':pages[0]==pages[1],'mount_identity_before':beforeid,'mount_identity_after':afterid,'exact_old_mount_occurrences':pages[0].count(beforeid),'exact_new_mount_occurrences':pages[1].count(afterid),'finite_mount_substitution_leaves_zero_other_deltas':True,'raw_ordered_svg_count_before':len(svgs[0]),'raw_ordered_svg_count_after':len(svgs[1]),'raw_ordered_svg_equal':svgs[0]==svgs[1],'svg_after_exact_same_mount_substitution_equal':True,'payload_json_deltas':payload_deltas,'scope':'Raw output hashes differ and remain unblessed. This exhaustive substitution only identifies the exact random mount-ID cause; no normalized output is produced or accepted.'};save(dest/'html-cause.json',htmlrow);htmlrows.append(htmlrow)
 if changed:
  assert len(changed)==1;key=changed[0];before=a[key];after=z[key];assert before['value']==after['value']and before['status']=='code_proven'and after['status']=='code_and_config';assert before['claim_proof']is None and after['claim_kind']=='relation'and after['claim_proof']['claim_kind']=='relation'
  oldir=load(old/'ir-before-render.json.gz');newir=load(new/'ir-before-render.json.gz');oldobs=oldir['extras']['config_access']['projection_obligations'];newobs=newir['extras']['config_access']['projection_obligations'];insertions=[(i,r)for i,r in enumerate(newobs)if newobs[:i]+newobs[i+1:]==oldobs];assert len(insertions)==1;index,ob=insertions[0];assert ob['expected_value_status_hash']==vh(after['value'],after['status'])and ob['target']['owner']+'.'+ob['target']['key']==key
  contexts=[load(p/'render-context-local.json.gz')['context']for p in(old,new)];events=[]
  for context,native in zip(contexts,(before,after)):
   er=[{'view':e['view'],'block_path':e['block_path'],'component':e['component'],'facts_projected':e['facts_projected'],'node_ids':e['node_ids'],'receipts':[r for r in e['receipts']if r['fact_id']==key]}for e in context['events']if key in e['facts_projected']['members']or any(r['fact_id']==key for r in e['receipts'])]
   receipts=[r for e in er for r in e['receipts']];assert len(receipts)==1 and receipts[0]['projection_kind']=='field'and receipts[0]['structural_target']==key.rsplit('.',1)[1]and receipts[0]['fact_value_status_hash']==vh(native['value'],native['status'])
   assert any(e['view']=='layer_map'and 'layer_map'in e['node_ids']['members']for e in er);assert all(not any(json.loads(ref['_record'])['fact_key']==key for ref in e['chip']['references'])for e in context['chip_events']);events.append(er)
  item={'key':key,'native_before':before,'native_after':after,'exact_added_consumption_index':index,'exact_added_consumption':ob,'existing_consumptions_identical_after_removing_exact_one_insertion':True,'render_events_before':events[0],'render_events_after':events[1],'graph_or_schedule_values_changed':False,'existing_layer_map_field_receipt_hash_now_matches_new_native_tier':True,'new_operation_receipt_count':0,'ordinary_checkpoint_class_default_chip_count':0,'source_cause':'Existing all-layer mechanism is retained; exact repeated constructor count now contributes its located checkpoint operand and relation proof. No new architectural value or card drawing.'};save(dest/'schedule-cause.json',item);special[slug]=item
 allmodels.append({'slug':slug,'native_delta_keys':changed,'native_recursive_delta_count':len(delta),'ir_before_delta_count':len(phase_deltas['ir-before-render.json.gz']),'raw_html_sha256':htmlrow['after_sha256']})
assert native_changes==[('bloom','decoder.attention.position_schedule'),('musicgen-small','decoder.attention.cross_attention_schedule')];assert phase_totals['ir-before-render.json.gz']==104
refs={}
for name,needle in [('model_unfolder/diagram.py','self._mount_id ='),('model_unfolder/adapters/transformer/parser.py','def _repeated_schedule_projection'),('model_unfolder/evidence/reader_claims.py','def repeated_count_operand'),('model_unfolder/evidence/position_linear_bias.py','return repeated_count_operand'),('model_unfolder/evidence/cross_attention_schedule.py','return repeated_count_operand'),('model_unfolder/renderers/html/views.py','position_schedule')]:
 p=F/name;lines=p.read_text().splitlines();hits=[i+1 for i,line in enumerate(lines)if needle in line];assert hits;refs[name]={'sha256':sha(p.read_bytes()),'needle':needle,'lines':hits}
summary={'verdict':'PASS_INDEPENDENT_SAVED34_DATA_AND_EXACT_CAUSE_AUDIT_NOT_ACCEPTANCE','archive_map_sha256':sha((A/'archive-map.json').read_bytes()),'archive_files_independently_verified_against_raw':590,'archive_raw_bytes':sum(r['bytes']for r in archive),'source_manifest_sha256':lane['source_manifest_sha256'],'lane':lane,'held_audit_manifest_sha256':sha((H/'manifest.json').read_bytes()),'held_audit_results':audit['totals'],'native_changed_keys':native_changes,'native_value_changes':0,'other37_native_documents_unchanged':True,'raw_IR_delta_counts_by_phase':dict(phase_totals),'raw_ir_causes':'Two native proof/provenance changes and exactly one newly recorded count consumption per model. All existing obligation rows/order remain after removing that exact insertion. Every raw leaf retained; index shifts not waived or deleted.','layers_and_structural_drawings_unchanged_all39':True,'raw_HTML_hash_changes':sum(not r['raw_equal']for r in htmlrows),'HTML_cause':'All39 raw pages differ only by their exact Diagram uuid4-derived mount ID. Exhaustive replacements and every raw diff retained; zero remaining SVG/geometry/text or payload differences after that finite identity substitution. No page is normalized or approved.','schedule_targets_live_qualified':2,'matrix_142_occurrence_closure':'Not established here; requires actual34 matrix. Native proof summaries are not reconstructed in this saved audit.','sparse_musicgen_limit':'This fleet uses the ordinary checkpoint with declared decoder.num_hidden_layers and code_and_config tier. It does not exercise the separate sparse MusicGen class_default missing-card-citation finding. Zero class_default chip on these ordinary two schedule facts is expected and does not close that sparse display gap.','source_references':refs,'browser_limit':'Serialized HTML/source inspection and original in-process events only; no browser execution, visibility/click/pixel verdict.','all_models':allmodels,'output_approval':False,'no_model_test_or_product_import_runtime_by_auditor':True}
save(O/'summary.json',summary);save(O/'all39-html-causes.json',htmlrows);(O/'audit-source.py').write_bytes(Path('/private/tmp/close34-independent-fleet.py').read_bytes());save(O/'manifest.json',{'files':{p.relative_to(O).as_posix():sha(p.read_bytes())for p in sorted(O.rglob('*'))if p.is_file()}});print(json.dumps({'manifest_sha256':sha((O/'manifest.json').read_bytes()),'native_changes':native_changes,'IR':dict(phase_totals),'HTMLcaused':len(htmlrows),'schemafindings':audit['totals']},indent=2))
