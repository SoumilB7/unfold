from pathlib import Path
import collections,hashlib,json,subprocess,tarfile
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');B=R/'verification/receipts/S9-A-schema-candidate-fff99e22';S=Path('/private/tmp/unfold-s9a-thirtyfirst/preservation-current31');A=B/'thirtyfirst/preservation-current31';O=B/'new-resume/reviews/current31-preservation-packet';O.mkdir(exist_ok=False)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(p):return json.loads(p.read_bytes())
def canon(v):return json.dumps(v,sort_keys=True,default=str).encode()
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
mp=A/'archive-map.json';assert sha(mp.read_bytes())=='d331469cdc898f60638a322e7c8d4046fef3531cfecd77894d79cd25b1de8337';mapping=load(mp);rows={v['path']:v for v in mapping['files']};assert len(rows)==len(mapping['files'])==20459
archive=A/'exact-capture.tar.gz';assert sha(archive.read_bytes())==mapping['archive_sha256'];count=0;size=0;seen=set()
with tarfile.open(archive,'r:gz') as tf:
 for member in tf:
  assert member.isfile() and member.name in rows and member.name not in seen
  raw=tf.extractfile(member).read();expected=rows[member.name];assert sha(raw)==expected['sha256'] and len(raw)==expected['bytes'];assert sha((S/member.name).read_bytes())==expected['sha256'];seen.add(member.name);count+=1;size+=len(raw)
assert seen==set(rows)
result=load(S/'result.json');report=load(S/'capture/report.json');worker=S/'capture/controller';actual=load(worker/'capture-result.json');expected_raw=(worker/'expected.executed.json').read_bytes();expected=json.loads(expected_raw);candidate=load(S/'capture/candidate-manifest.UNAPPROVED.json')
pairs={}
for label,left,right in [('outer',S/'pins-before.json',S/'pins-after.json'),('child',S/'child-pins-before.json',S/'child-pins-after.json'),('observer',worker/'pins-before.json',worker/'pins-after.json')]:
 a,b=load(left),load(right);assert a==b;pairs[label]={'equal':True,'before_sha256':sha(left.read_bytes()),'after_sha256':sha(right.read_bytes())}
outer=load(S/'pins-before.json');assert outer['source_manifest_sha256']=='153982d5541dbd6aa50c2bdfe3dee2fa9c4b92b93f7c26d11c203de1dc1d1a01';assert load(S/'child-pins-before.json')==outer
held=B/'new-resume/current31-preservation-capture';held_manifest=load(held/'manifest.json');assert sha((held/'manifest.json').read_bytes())=='3c0e3d4a45ac4303c0f13369ca3a73a04c649f19712d1778ab16160120ea23cf'
for name,h in outer['scripts'].items():assert sha((held/name).read_bytes())==h
assert sha((worker/'capture_preservation.executed.py').read_bytes())==outer['scripts']['capture_preservation.py']
assert sha(expected_raw)==outer['frozen_inputs']['expected_manifest']==load(worker/'pins-before.json')['expected_manifest']
tree=R/'.claude/worktrees/verify-s9-a-thirtyfirst';assert subprocess.check_output(['git','show',outer['head']+':tests/preservation_expected_manifest.json'],cwd=tree)==expected_raw
assert result['before']==result['after'] and result['artifacts_before']==result['artifacts_after'] and result['pins_equal']
assert result['returncode']==1 and not result['preservation_passed'] and result['packet_complete']
assert report['packet_complete'] and not report['raw_preservation_passed'] and actual['capture_complete'] and not actual['capture_errors']
assert len(actual['collection'])==len(set(actual['collection']))==report['collected_count']==report['reported_test_count']==52
assert set(report['original_test_reports'])==set(actual['collection'])
outcomes=collections.Counter(v['outcome'] for rows in report['original_test_reports'].values() for v in rows);assert outcomes=={'passed':23,'failed':29}
assert expected['witness_count']==candidate['witness_count']==29 and candidate['authority']=='UNAPPROVED_CANDIDATE_ONLY'
assert len(report['cases'])==29 and set(expected['witnesses'])==set(report['witness_counts']) and all(n==1 for n in report['witness_counts'].values())
surfaces=('ir','expanded','params','html_meta','gallery','ledgers','sable');counts=collections.Counter();witnesses={};historical=collections.Counter();view_count=0
for c in report['cases']:
 slug=c['slug'];exp=expected['witnesses'][slug];assert c['original_exception']is None and not c['capture_errors'] and c['input_matches_expected'];assert c['call_counts']=={'canonical_surfaces':1,'gallery_witness':1,'_view_hashes':1}
 def artifact(ref):
  p=worker/ref['path'];raw=p.read_bytes();assert sha(raw)==ref['sha256'] and len(raw)==ref['bytes'];return json.loads(raw)
 docs=artifact(c['actual_surfaces']);input_doc=artifact(c['input']);assert sha(canon({'config':input_doc.get('config'),'source':input_doc.get('source','local')}))==exp['input_sha256']
 assert set(docs)==set(exp['surfaces'])==set(surfaces)
 calls={v['name']:artifact(v['artifact']) for v in c['calls']};captured= calls['canonical_surfaces'];captured['gallery']=calls['gallery_witness'];assert docs==captured
 actual_hashes={s:sha(canon(docs[s])) for s in surfaces};assert actual_hashes==c['surface_sha256']==candidate['witnesses'][slug]['surfaces']
 assert candidate['witnesses'][slug]==c['candidate'];assert c['views']==calls['_view_hashes']==candidate['witnesses'][slug]['views'];assert c['views_equal_expected'] and c['views']==exp['views'] and c['view_deltas']==[];view_count+=len(c['views'])
 changed=sorted(s for s in surfaces if actual_hashes[s]!=exp['surfaces'][s]);assert changed==sorted(c['surface_deltas'])==['html_meta','ir','ledgers','sable']
 assert sorted(c['original_findings'])==sorted(f'{slug}/{s}: hash mismatch' for s in changed)
 assert c['renders'] and any(row['is_original_html_surface'] for row in c['renders'])
 for s,entry in c['surface_comparisons'].items():
  assert entry['actual_sha256']==actual_hashes[s] and entry['expected_sha256']==exp['surfaces'][s]
  historical[entry['baseline_document_status']]+=1
  if not entry['stored_document_matches_expected']:assert entry['document_delta_authority']=='HISTORICAL_ONLY_NOT_THE_COMMITTED_EXPECTED_SURFACE'
 counts.update(changed)
 witnesses[slug]={'changed_surface_count':len(changed),'changed_surfaces':changed,'unchanged_surfaces':sorted(set(surfaces)-set(changed)),'view_sequence_equal':True,'expected_view_count':len(exp['views']),'actual_view_count':len(c['views']),'surface_hashes':{s:{'committed_expected_sha256':exp['surfaces'][s],'actual31_sha256':actual_hashes[s],'equal':actual_hashes[s]==exp['surfaces'][s]} for s in surfaces},'historical_document_gaps':[s for s,v in c['surface_comparisons'].items() if not v['stored_document_matches_expected']],'original_findings':c['original_findings']}
assert historical=={'EXACT_EXPECTED_DOCUMENT':85,'HISTORICAL_DOCUMENT_HASH_MISMATCH':118}
summary={'verdict':'PASS_PACKET_COMPLETENESS_AND_AUTHORITY_PRESERVATION_FAILS_UNAPPROVED','scope':'Independent stdlib saved-data audit only; no model/test/product imports, no active artifact writes. Full leaf cause comparison deferred until baseline recovery matches committed surface hash.', 'archive':{'map_sha256':sha(mp.read_bytes()),'archive_sha256':mapping['archive_sha256'],'files_verified_against_archive_and_raw':count,'raw_bytes':size},'brackets':pairs,'source_manifest_sha256':outer['source_manifest_sha256'],'held_observer_manifest_sha256':sha((held/'manifest.json').read_bytes()),'committed_expected_manifest_sha256':sha(expected_raw),'original_test_outcomes':dict(outcomes),'original_test_count':52,'witness_count':29,'surface_comparisons':203,'changed_surface_comparisons':sum(counts.values()),'unchanged_surface_comparisons':203-sum(counts.values()),'changed_counts_by_surface':dict(counts),'all29_view_sequences_equal':True,'total_distinct_view_rows':view_count,'historical_document_status_counts':dict(historical),'capture_complete':True,'preservation_passed':False,'output_approved':False,'witnesses':witnesses,'limitations':['All29 witnesses differ on ir/html_meta/ledgers/sable. Expanded/params/gallery and established visual view sequences match committed hashes.', 'A complete packet does not prove these changes are correct or chips-only. Full evidence-level cause attribution remains pending baseline recovery.', '118 historical surface documents do not match authoritative hashes and are not used as expected leaf data in this audit.', 'Candidate version metadata is explicitly copied expected context, not a fresh measured package-version claim.']}
save(O/'summary.json',summary)
lines=['| Witness | Changed canonical surfaces | Changed count | Visual views unchanged |','|---|---|---:|---:|']
for slug,w in sorted(witnesses.items()):lines.append(f"| {slug} | {', '.join(w['changed_surfaces'])} | {w['changed_surface_count']} | {w['actual_view_count']} |")
(O/'witness-summary.md').write_text('\n'.join(lines)+'\n')
(O/'audit-source.py').write_bytes(Path('/private/tmp/audit-current31-preservation-packet.py').read_bytes())
save(O/'manifest.json',{'files':[{'path':p.name,'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(O.iterdir()) if p.is_file()]})
print(json.dumps({'summary_sha256':sha((O/'summary.json').read_bytes()),'manifest_sha256':sha((O/'manifest.json').read_bytes()),'archive_files':count,'raw_bytes':size,'surface_changes':dict(counts),'distinct_view_rows':view_count,'tests':dict(outcomes)},indent=2))
