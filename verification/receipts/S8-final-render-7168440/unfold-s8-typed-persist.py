from pathlib import Path
import json,gzip,hashlib,shutil,html
base=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer');run=Path('/private/tmp/unfold-s8-typed-phase-7168440');receipt=base/'unfold-pkg/verification/receipts/S8-final-render-7168440';pages=base/'z-docs/12-design/S8/final-7168440'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
assert not receipt.exists() and not pages.exists();receipt.mkdir();pages.mkdir()
records=sorted(run.glob('*/result.json'));assert len(records)==11
control=read(run/'raw-control-identity.json');assert len(set(control.values()))==1
audits=read(run/'actual-delta-audit.json');assert len(audits)==11
rows=[]
for p in records:
 src=p.parent;name=src.name;r=read(p);assert r['status']=='PASS' and not any(r['checks'].values());out=receipt/name;out.mkdir();proofdir=Path(r['summary_proof_dir']);target=out/'summary-proof';target.mkdir()
 for file in src.iterdir():
  if file.suffix=='.json' and file.name not in ('old-pure-projection.json','new-pure-projection.json','ir.json','render-input.json'):shutil.copy2(file,out/file.name)
  elif file.name=='render-events.json.gz':shutil.copy2(file,out/file.name)
  elif file.name in ('old-pure-projection.json','new-pure-projection.json','ir.json','render-input.json','page.html'):(out/(file.name+'.gz')).write_bytes(gzip.compress(file.read_bytes(),mtime=0))
 for file in proofdir.glob('*.json'):
  if file.name in ('proofs.json','request.json'):(target/(file.name+'.gz')).write_bytes(gzip.compress(file.read_bytes(),mtime=0))
  else:shutil.copy2(file,target/file.name)
 shutil.copy2(src/'page.html',pages/(name+'.html'))
 events=read(src/'event-deltas.json');audit=read(src/'actual-delta-audit.json');deltas=read(src/'ir-deltas.json')
 row={'case':name,'actual_html':str(pages/(name+'.html')),'actual_html_sha256':sha(src/'page.html'),'prior_5227_html_sha256':r['prior_html_sha256'],'ir_sha256':sha(src/'ir.json'),'render_input_sha256':sha(src/'render-input.json'),'full_event_uncompressed_sha256':hashlib.sha256(gzip.decompress((src/'render-events.json.gz').read_bytes())).hexdigest(),'source_manifest_sha256':sha(src/'source-before.json'),'input_and_proof_pin_sha256':sha(src/'input-pins.json'),'summary_proof_result_sha256':sha(proofdir/'result.json'),'construction_summary':deltas['construction_summary'],'component_entry':deltas['component_entry'],'checks':r['checks'],'svg_delta_count':len(r['svg_deltas']),'receipt_events_added':len(events['added']),'receipt_events_removed':len(events['removed']),'exact_html_delta_audit':name+'/actual-delta-audit.json','historical_fact_case':r['fact_case'],'historical_base_5227_case':r['base_case'],'elapsed_seconds':r['elapsed_seconds'],'blessed':False};rows.append(row)
for source in ('/private/tmp/unfold-s8-final-typed-render.py','/private/tmp/unfold-s8-typed-campaign.py','/private/tmp/unfold-s8-typed-delta-audit.py',__file__):shutil.copy2(source,receipt/Path(source).name)
shutil.copy2(run/'raw-control-identity.json',receipt/'raw-control-identity.json');shutil.copy2(run/'actual-delta-audit.json',receipt/'incremental-output-ledger.json')
summary={'production_commit':'7168440c81a3e79b1e220437e2f9fa77e25bc296','phase':'Existing saved case evidence; exact newly rederived per-case construction summary; bounded projection metadata/label/typed-entry migration; real Diagram rendering. No new model construction or execution.','case_count':len(rows),'cases':rows,'raw_ordinary_rewrite_unchanged':control,'all_seven_checks_pass':True,'all_actual_svg_bytes_unchanged_from_5227':True,'canonical_facts_and_quantities_unchanged':True,'all_prior_events_retained':True,'historical_differential_dispositions_unchanged':True,'known_limitations':['DDPM six unbound bookends and open stage-target connections remain.','Bounded source-proven ResNet fragment does not prove the entire residual computation.','Native SVG inspection history applies by exact SVG-byte equality; no new browser/pixel inspection of HTML card prose was performed.'],'blessed':False}
write(receipt/'summary.json',summary);write(pages/'manifest.json',summary);shutil.copy2(receipt/'incremental-output-ledger.json',pages/'incremental-output-ledger.json')
(receipt/'README.md').write_text('''# Final saved-evidence render phase:7168440

All eleven actual pages pass the five existing click, tree, ID and stroke validators, label lint and the projection audit. Seven SDXL controls, SD-v1-4 ordinary and the three extra component witnesses retain exact canonical fact bytes and exact parameter records. This phase never rebuilds a model, executes a model, reinterprets missing mechanisms or blesses outputs.

Each case transports its own final rederived ConstructionSummary after verifying the three full qualified values, claim kinds, original input/inventory/fact hashes, exact summary fields and production manifests. Full reproof records and archive-byte checks are retained per case. Historical scratch/source/model records remain at their original pins. The three proof source reads are separately recorded by the executor; they are not a new full architecture parse.

Old5227 and current pure projectors consume the same saved values in separate source-pinned subprocesses. The only transferred changes are enumerated activation operation labels, fact_display_lines metadata for already present exact chips, the optional typed component interface and the qualified construction summary. Reversing these exact transfers must recover the complete original IR. Empty outer handoffs used for the bounded comparison never replace the supplied complete pipeline. Typed component-entry input IDs must match the original producer-owned scope and exact input IDs; full supplied SDXL has no component entry.

Every actual SVG is byte-identical to5227. The complete actual HTML equals the prior page after only the listed activation card-title capitalization and, on component-only cases, explicit DENOISER COMPONENT heading/subtitle replacements. No HTML bytes are rewritten to achieve this result: this equality is a diagnostic against actual Diagram output. The missing-evidence page remains byte-identical. Warnings and quantities remain exact.

All prior render events remain; the new events are only exact displayed default/spatial chips and FFN mechanisms whose returned SVG contains their declared operation nodes. These are enumerated additions, not an event-equality claim. Each case records the complete new event sequence and added/removed event ledger. The original old/new architecture differential remains unchanged in authority and owner disposition; this receipt is its incremental presentation/receipt delta only.

Ordinary, supported equivalent rewrite and unchanged scratch actual HTML bytes are identical at the final renderer. Exact hashes are in summary.json and raw-control-identity.json. Previous phases and failure artifacts remain preserved. DDPM still has six unbound bookend connections and open stage-target limitations; the bounded ResNet claim remains a member-connection fragment. Native diagram review carries forward solely through exact SVG equality; HTML card prose changes are artifact checks, not new browser pixel review.

Decision page: root z-docs/12-design/S8/final-7168440/index.html. Executor owns the independent final coverage/matrix and broad gates. Nothing is automatically approved or blessed here.
''')
links=''.join('<li><a href="'+r['case']+'.html">'+r['case']+'</a> — seven checks pass; '+str(r['receipt_events_added'])+' exact receipt events added; SVGs unchanged.</li>' for r in rows)
(pages/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S8 final actual pages ·7168440</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;line-height:1.6}li{margin:10px 0}code{overflow-wrap:anywhere}</style><h1>S8 final actual pages ·7168440</h1><p>Eleven saved-evidence replays with case-specific qualified quantities. Click, tree, ID, stroke, label and projection checks pass. All diagrams retain the exact previously inspected SVG bytes. Existing card facts now have precise render receipts; component-only pages are labelled as components.</p><p>Historical model/source runs retain their original pins. No model rerun or output blessing. DDPM still has six unbound bookend connections and open stage-target limitations.</p><p>Ordinary/rewrite/unchanged actual HTML SHA-256: <code>'+next(iter(control.values()))+'</code>.</p><ul>'+links+'</ul><p><a href="manifest.json">Source, proof and output manifest</a> · <a href="incremental-output-ledger.json">Exact incremental output ledger</a> · <a href="../final-5227e9c/index.html">Preserved5227 pages</a> · <a href="../final-c38b008/render-phase/index.html">Original source controls</a></p>')
write(receipt/'artifact-sha256.json',{str(p.relative_to(receipt)):sha(p) for p in sorted(receipt.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'})
print(json.dumps({'receipt':str(receipt),'decision_page':str(pages/'index.html'),'summary_sha256':sha(receipt/'summary.json'),'index_sha256':sha(pages/'index.html')}))
