from pathlib import Path
import json,gzip,hashlib,shutil,sys,html
base=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer'); run=Path('/private/tmp/unfold-s8-solid-f7baef5'); receipt=base/'unfold-pkg/verification/receipts/S8-solid-stroke-f7baef5'; page=base/'z-docs/12-design/S8/final-f7baef5';receipt.mkdir(exist_ok=True);page.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,'/private/tmp/unfold-s8-solid-review-f7baef5')
from model_unfolder.preview import svg_to_png
rows=[]
for cp in sorted(run.glob('*/comparison.json')):
 case=cp.parent;name=case.name;b=json.loads((case/'before/result.json').read_text());a=json.loads((case/'after/result.json').read_text());c=json.loads(cp.read_text());src=Path(a['source_case']);prior=src/'page.html'
 if name in ('sdxl-refiner','sdxl-inpainting','ddpm-cifar10'):prior=Path('/private/tmp/unfold-s8-component-render-96d0c1e')/name/'page.html'
 assert prior.read_bytes()==(case/'before/page.html').read_bytes()
 target=receipt/name;target.mkdir(exist_ok=True)
 for phase in ('before','after'):
  out=target/phase;out.mkdir(exist_ok=True)
  for f in ('result.json','render-events.json.gz'):shutil.copy2(case/phase/f,out/f)
  (out/'page.html.gz').write_bytes(gzip.compress((case/phase/'page.html').read_bytes(),mtime=0))
 for f in ('comparison.json','denoiser-before.svg','denoiser-after.svg','before.log','after.log'):shutil.copy2(case/f,target/f)
 shutil.copy2(case/'after/page.html',page/f'{name}.html');shutil.copy2(cp,page/f'{name}-comparison.json');shutil.copy2(case/'denoiser-after.svg',page/f'{name}-denoiser.svg')
 if name in ('sdxl-ordinary','sd14-ordinary','sdxl-refiner','sdxl-inpainting','ddpm-cifar10'):
  svg_to_png((case/'denoiser-after.svg').read_text(),str(page/f'{name}-denoiser.png'),scale=1,highlight_clickable=True)
 row={'case':name,'prior_actual_page':str(prior),'prior_actual_sha256':sha(prior),'prior_replay_byte_identical':True,'final_actual_page':str(page/f'{name}.html'),'final_actual_sha256':a['actual_html_sha256'],'original_parse_artifact_pins':a['source_artifact_pins'],'prior_renderer_commit':b['production_commit'],'final_renderer_commit':a['production_commit'],'render_event_count':a['render_event_count'],'before_render_event_sha256':b['render_event_sha256'],'after_render_event_sha256':a['render_event_sha256'],'ir_warnings_quantities_exact':True,'full_render_events_identical':True,'only_svg_changed':c['changed_svg_labels'],'removed_dash_attributes':c['removed_dash_attributes'],'dotted_boundary_findings_before':c['before_dotted_boundary_count'],'dotted_boundary_findings_after':0,'dotted_arrow_findings_after':0,'blessed':False};rows.append(row)
raw=json.loads((run/'raw-control-identity.json').read_text());assert len(set(raw.values()))==1
summary={'scope':'Actual Diagram re-render from captured exact inputs; no model or evidence-reader rerun. Original differential classifications remain untouched.','renderer_commit':'f7baef5289e538f7284f1f1542f5897a780429ba','case_count':len(rows),'cases':rows,'raw_ordinary_rewrite_unchanged':raw,'all_checks_pass':True,'blessed':False};(receipt/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(page/'manifest.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
for f in ('worker.py','compare.py','remaining.py','persist.py','raw-control-identity.json'):shutil.copy2(run/f,receipt/f)
text='''# Solid-stroke correction: final actual renderer phase

All eleven exact saved-input replays passed at f7baef5289e538f7284f1f1542f5897a780429ba. The prior renderer was96d0c1e. Each before-page is byte-identical to its already published actual page (the extra trio uses its previous96 component-label phase). Original pages, parse/source pins, condition reports and differential classifications remain preserved. This phase neither reclassifies a delta nor blesses an output.

Only the denoiser SVG changes: remove the two exact dash-array attributes and replace the three existing conditional-target captions with accurate identity-link wording. Every other HTML byte is accounted for by those exact SVG replacements. All card IDs, route endpoints and attributes remain identical apart from the removed dash attribute. Identity links have no arrowheads. Both existing no-dotted validators report zero on every final actual page; no validator was changed.

Fresh-original-JSON equality, typed warning reconstruction, exact captured Diagram input and parameter-record equality protect every replay. Original IR/facts/qualified-facts/result hashes remain unchanged. Full canonical RenderEvent sequences are persisted before and after and are byte-identical for every case; only unordered set values are sorted, and all list/tuple order is preserved. This is the observed rendering-event premise for downstream reconciliation pin refresh, not a new runtime observation or mechanism proof.

Final ordinary/rewrite/unchanged actual HTML bytes are identical: 86bb0629771f875bdba3e1fb219c403160938254c4ec2ee5199baaec562bc5bf. Per-case original/new page hashes, source pins, event counts/hashes and exact comparisons are in summary.json and each case directory. Production manifests include Python/YAML sources before and after each real render and are identical within each replay.

Root visually inspected both corrected actual SDXL and SD-v1-4 overview PNGs and accepted this bounded stroke/caption correction: solid frames, labels fit, no added arrowheads and no newly observed crossings. This is not an exhaustive visual claim. DDPM retains its six unresolved bookend connections; the previous scope/shape and mechanism limitations are unchanged. The previous96 entry-label correction remains present for the extra trio.

Actual pages: root z-docs/12-design/S8/final-f7baef5/index.html. Historical six-condition, quantity, claim-trace and differential reports remain in S8-final-c38b008 and S8-generalization-c38b008; exact page equivalence except the enumerated SVG stroke/caption delta transports their unchanged content, without rewriting their original attribution. No model or pytest lane ran for this receipt.
''';(receipt/'README.md').write_text(text)
links=''.join(f'<li><a href="{r["case"]}.html">{r["case"]}: actual full page</a> · <a href="{r["case"]}-denoiser.svg">denoiser SVG</a> · <a href="{r["case"]}-comparison.json">exact comparison</a></li>' for r in rows)
(page/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S8 final solid-stroke renderer phase</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;line-height:1.6}li{margin:10px 0}code{overflow-wrap:anywhere}</style><h1>S8 · final actual renderer phase f7baef5</h1><p>Eleven saved-input actual Diagram replays. No models rerun; no outputs blessed. Existing dotted-boundary and dotted-arrow checks pass. Only denoiser stroke attributes and conditional identity captions changed. Exact facts, quantities, cards and render events are retained.</p><p>Ordinary, equivalent rewrite and unchanged scratch control have identical raw HTML SHA-256: <code>'+next(iter(raw.values()))+'</code>.</p><p>The extra trio retains its previous entry-label correction. DDPM still has six unbound bookend connections; geometry does not qualify them.</p><ul>'+links+'</ul><p><a href="manifest.json">Per-case hashes and source/event pins</a> · <a href="../final-c38b008/render-phase/index.html">Preserved original condition pages and reports</a></p>')
manifest={str(p.relative_to(receipt)):sha(p) for p in sorted(receipt.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'};(receipt/'artifact-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'receipt':str(receipt),'page':str(page/'index.html'),'cases':len(rows),'summary_sha256':sha(receipt/'summary.json'),'index_sha256':sha(page/'index.html')}))
