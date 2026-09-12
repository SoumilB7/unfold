from pathlib import Path
import collections,difflib,gzip,hashlib,json,re
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');B=R/'verification/receipts/S9-A-schema-candidate-fff99e22';A=B/'thirtysixth/current-pages';RAW=Path('/private/tmp/unfold-s9a-thirtysixth/current-pages');OLD=Path('/private/tmp/unfold-s9a-thirtyfourth/current-pages/capture');H=B/'new-resume/reviews/current36-saved-audit';O=B/'new-resume/reviews/current36-saved-independent-event';O.mkdir(exist_ok=False);F=R/'.claude/worktrees/verify-s9-a-thirtysixth'
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

assert sha((A/'archive-map.json').read_bytes())=='073125cc0b3c75b745e0602cc93a59429e4b9a2754b37f6922444f2af038a674';archive=load(A/'archive-map.json')['files'];assert len(archive)==590
for row in archive:
 p=B/row['archive'];q=Path(row['source']);assert q.relative_to(RAW)==p.relative_to(A);raw=p.read_bytes();assert len(raw)==row['bytes']and sha(raw)==row['sha256']and raw==q.read_bytes()
assert load(A/'pins-before.json')==load(A/'pins-after.json');lane=load(A/'result.json');assert lane['passed']and lane['pins_equal']and lane['before']==lane['after']=='88c235ca5e1cde3ca9b20aa9f6f133bd4ced546e8ea26757d57bb972e5d6edbe'and lane['artifacts_before']==lane['artifacts_after']=='5aed5480485a877c5393ec4eb97be56cf2709327baa35ede0da54884029bd97e'
assert sha((H/'manifest.json').read_bytes())=='73b2f6836989de7f5c9535a51d545de635db79cf105c5ef3f5b313874f70789a';assert all(sha((H/r['path']).read_bytes())==r['sha256']for r in load(H/'manifest.json')['files'])
fleet=load(A/'capture/result.json');assert fleet['captured_count']==39 and fleet['failed_count']==0 and len(fleet['models'])==39;audit=load(H/'results/summary.json');assert len(audit['models'])==39
for k in ('metadata_findings','missing_rendered_paths','unexpected_rendered_paths','class3_records','capture_census_findings'):assert audit['totals'][k]==0
phase_totals=collections.Counter();htmlrows=[];allmodels=[];citations=[];event_token_deltas=0
for model in fleet['models']:
 slug=model['slug'];new=A/'capture'/slug;old=OLD/slug;dest=O/slug;dest.mkdir()
 for name,digest in model['artifact_sha256'].items():assert sha((new/name).read_bytes())==digest
 a=load(old/'declarations.json');z=load(new/'declarations.json');delta=diff(a,z);save(dest/'native-deltas.json',delta);assert delta==[]
 phase_deltas={}
 for name in ('ir-before-render.json.gz','ir-after-render.json.gz','diagram-ir-before-render.json.gz','diagram-ir-after-render.json.gz'):
  before=load(old/name);after=load(new/name);d=diff(before,after);phase_deltas[name]=d;phase_totals[name]+=len(d)
  for row in d:
   path=row['path'];assert len(path)==5 and path[0]=='layers'and path[2]=='blocks'and path[4]=='source_fact_keys'
   assert not row['before_present']and row['after_present']and len(row['after'])==1
   block=after['layers'][path[1]]['blocks'][path[3]];key=row['after'][0]
   expected={'bloom':('attn','decoder.attention.position_schedule'),'musicgen-small':('cross_attn','decoder.attention.cross_attention_schedule')}
   assert slug in expected and (block['id'],key)==expected[slug]and block['kind']=='attention'and block['view']=='attention'
   assert key in z and z[key]['claim_kind']=='relation'and z[key]['claim_proof']is not None
  if name=='ir-before-render.json.gz':citations.extend({'slug':slug,'path':r['path'],'fact_key':r['after'][0]}for r in d)
 save(dest/'all-phase-ir-deltas.json',phase_deltas)
 contexts=[load(p/'render-context-local.json.gz')['context']for p in(old,new)];eventsdelta=diff(contexts[0]['events'],contexts[1]['events']);save(dest/'raw-render-event-deltas.json',eventsdelta)
 assert all(len(r['path'])==4 and r['path'][1]=='receipts'and r['path'][3]=='context_token'for r in eventsdelta)
 event_token_deltas+=len(eventsdelta)
 rowaudit=load(H/'results'/f'{slug}.json');assert {r['path']:r['sha256']for r in rowaudit['inputs']}=={p.name:sha(p.read_bytes())for p in sorted(new.iterdir())if p.is_file()}
 pages=[gzip.decompress((p/'page.html.gz').read_bytes()).decode()for p in(old,new)];ids=[re.findall(r'<div id="(uf-[0-9a-f]{10})" class="uf-root">',page)for page in pages];assert all(len(v)==1 for v in ids);beforeid,afterid=ids[0][0],ids[1][0];assert pages[0].replace(beforeid,afterid)==pages[1]
 rawdiff=''.join(difflib.unified_diff(pages[0].splitlines(True),pages[1].splitlines(True),fromfile='actual34/'+slug+'.html',tofile='actual36/'+slug+'.html'));(dest/'raw-html.diff.gz').write_bytes(gzip.compress(rawdiff.encode(),mtime=0));svgs=[re.findall(r'<svg\b.*?</svg>',page,re.S)for page in pages];assert svgs[0] and [s.replace(beforeid,afterid)for s in svgs[0]]==svgs[1]
 payloads=[re.findall(r'<script\b[^>]*data-uf-card-payload[^>]*>(.*?)</script>',page,re.S)for page in pages];assert len(payloads[0])==len(payloads[1]);payload_deltas=[diff(json.loads(x),json.loads(y))for x,y in zip(*payloads)];assert not any(payload_deltas)
 htmlrow={'slug':slug,'before_sha256':sha(pages[0].encode()),'after_sha256':sha(pages[1].encode()),'before_bytes':len(pages[0].encode()),'after_bytes':len(pages[1].encode()),'raw_equal':pages[0]==pages[1],'mount_identity_before':beforeid,'mount_identity_after':afterid,'exact_old_mount_occurrences':pages[0].count(beforeid),'exact_new_mount_occurrences':pages[1].count(afterid),'finite_mount_substitution_leaves_zero_other_deltas':True,'raw_ordered_svg_count_before':len(svgs[0]),'raw_ordered_svg_count_after':len(svgs[1]),'raw_ordered_svg_equal':svgs[0]==svgs[1],'svg_after_exact_same_mount_substitution_equal':True,'payload_json_deltas':payload_deltas,'scope':'Raw output hashes differ and remain unblessed. Exhaustive finite substitution identifies the exact random mount-ID cause only; no normalized output is produced or accepted.'};save(dest/'html-cause.json',htmlrow);htmlrows.append(htmlrow)
 allmodels.append({'slug':slug,'native_delta_count':len(delta),'IR_deltas_by_phase':{k:len(v)for k,v in phase_deltas.items()},'render_receipt_local_token_deltas':len(eventsdelta),'raw_html_sha256':htmlrow['after_sha256']})
assert len(citations)==94 and collections.Counter(r['slug']for r in citations)=={'bloom':70,'musicgen-small':24};assert set(phase_totals.values())=={94}
refs={}
for name,needle in [('model_unfolder/diagram.py','self._mount_id ='),('model_unfolder/adapters/transformer/parser.py','_repeated_schedule_cards ='),('model_unfolder/evidence/reader_claims.py','default_spellings =')]:
 p=F/name;lines=p.read_text().splitlines();hits=[i+1 for i,line in enumerate(lines)if needle in line];assert hits;refs[name]={'sha256':sha(p.read_bytes()),'needle':needle,'lines':hits}
summary={'verdict':'PASS_INDEPENDENT_SAVED36_DATA_AND_EXACT_CAUSE_AUDIT_NOT_ACCEPTANCE','archive_map_sha256':sha((A/'archive-map.json').read_bytes()),'archive_files_independently_verified_against_raw':590,'archive_raw_bytes':sum(r['bytes']for r in archive),'source_manifest_sha256':lane['source_manifest_sha256'],'lane':lane,'held_audit_manifest_sha256':sha((H/'manifest.json').read_bytes()),'held_audit_results':audit['totals'],'native_changed_keys':[],'native_values_statuses_claim_kinds_and_proof_summaries_unchanged_all39':True,'raw_IR_delta_counts_by_phase':dict(phase_totals),'raw_IR_causes':'Exactly94 added existing-card source_fact_keys lists:70 Bloom attn cards cite the unchanged qualified position_schedule;24 MusicGen cross_attn cards cite the unchanged qualified cross_attention_schedule. No other IR field/value/count/order changed in any of the four captured phases. Every raw leaf retained.','exact_card_citations':citations,'raw_render_event_delta_cause':'Only original process-local receipt context_token differs across runs. Every raw event delta is retained; no other event/receipt facts, node IDs, projection kind, hash, ordering or membership differs. No local token is reused as proof.','raw_receipt_context_token_delta_count':event_token_deltas,'raw_HTML_hash_changes':sum(not r['raw_equal']for r in htmlrows),'HTML_cause':'All39 raw pages differ only by their exact Diagram uuid4-derived mount ID. Exhaustive replacements and every raw diff retained; zero remaining SVG/geometry/text or payload differences. No page normalized or approved.','matrix_limit':'94 existing-card citations are not proof of occurrence placement. Actual36 matrix is still running;142 original schedule-related finding closures remain unclaimed by this page audit.','sparse_limit':'Fleet contains ordinary declared-count checkpoints. Separate actual36 sparse MusicGen24 and Bloom2 pages are archived/audited by placement; this fleet does not independently exercise those defaults.','source_references':refs,'browser_limit':'Serialized HTML/source inspection and original captured events only; no browser, click, visibility or pixel verdict.','all_models':allmodels,'output_approval':False,'no_model_test_or_product_import_runtime_by_auditor':True}
save(O/'summary.json',summary);save(O/'all39-html-causes.json',htmlrows);(O/'audit-source.py').write_bytes(Path('/private/tmp/close36-independent-fleet.py').read_bytes());save(O/'manifest.json',{'files':{p.relative_to(O).as_posix():sha(p.read_bytes())for p in sorted(O.rglob('*'))if p.is_file()}});print(json.dumps({'manifest_sha256':sha((O/'manifest.json').read_bytes()),'IR':dict(phase_totals),'HTMLcaused':len(htmlrows),'receipt_token_deltas':event_token_deltas,'schemafindings':audit['totals']},indent=2))
