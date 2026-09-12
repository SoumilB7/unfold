from pathlib import Path
import json,html,re,hashlib,shutil
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer');pages=root/'z-docs/12-design'
receipt=root/'unfold-pkg/verification/receipts/S9-A-schema-candidate-fff99e22'
record=json.loads((receipt/'thirtyfourth/current-pages/capture/result.json').read_text());rows=record['models']
assert len(rows)==39 and record['failed_count']==0
out=receipt/'new-resume/current34-decision-page';out.mkdir(exist_ok=False)
family_paths=[pages/('S9-'+f)/'index.html' for f in sorted({r['page_family'] for r in rows})]
targets=[pages/'S9-schema/index.html',*family_paths]
for p in targets:
 d=out/'before'/p.relative_to(root);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
e=html.escape
style='body{font:16px/1.55 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#182033;background:#f8fafc}a{color:#145ab5}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde3eb}th{font-size:13px}aside{padding:16px;border-left:4px solid #d99d29;background:#fff8e7;margin:24px 0}small{color:#596579}nav{display:flex;gap:20px;flex-wrap:wrap}'
def page(title,body):return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(title)+'</title><style>'+style+'</style></head><body>'+body+'</body></html>\n'
proofs=sum(r['validated_proof_count'] for r in rows)
body='<h1>S9-A · candidate 34 review pages</h1><p>Actual pages from frozen candidate 34 on <code>audio-composite-support</code>, based on <code>fff99e22</code>.</p><aside><strong>In review · outputs unblessed.</strong> All 39 ordinary corpus pages passed automated presentation checks; '+str(proofs)+' supplied fact proofs validated. 447 focused checks pass. Candidate 34 matrix and complete gate have not run; final preservation, output approval and browser review remain outstanding.</aside>'
body+='<p>The separate sparse MusicGen witness retains 24 layers and a native <code>class_default</code> schedule with the correct final consumption hash, but its actual page has no visible class-default chip. That visibility gap is being corrected. Sparse Bloom also exposes a separate count gap: omitting checkpoint <code>n_layer=70</code> yields zero layers despite the genuine prepared default of 2. The prepared correction must produce 2, not infer 70 from model identity. MusicGen 24→24 is the structure-preserving counterpart. These sparse failures are not hidden by the ordinary-page checks.</p>'
body+='<p>The latest completed matrix is candidate 32: 39 models / 44,088 occurrences; 5,065 rendered, 4,740 grouped, 1,463 nonarchitectural and 32,820 unresolved. Its 623 unstamped findings are 142 schedule findings plus 481 score-formula findings. Candidate 34 implements schedule qualification, but no new matrix count is asserted here.</p>'
body+='<nav><a href="../../11-execution/S9-A.md">Completion sheet</a><a href="current-thirtysecond.html">Previous actual pages (32)</a><a href="matrix-twentysixth.html">Earlier matrix (26)</a><a href="../S9-unet/source-stamp/index.html">Held S10 source-stamp proposal</a></nav>'
body+='<h2>Actual ordinary model pages</h2><p>Proof counts measure supplied proofs checked, not architecture completeness or browser acceptance.</p><table><thead><tr><th>Model</th><th>Family</th><th>Fact proofs checked</th><th>Presentation findings</th><th>HTML size</th></tr></thead><tbody>'
for row in sorted(rows,key=lambda x:(x['page_family'],x['slug'])):
 body+='<tr><td><a href="../'+e(row['published_page'])+'">'+e(row['slug'])+'</a></td><td>'+e(row['page_family'])+'</td><td>'+str(row['validated_proof_count'])+'</td><td>'+str(len(row['presentation_findings']))+'</td><td>'+f"{row['page_bytes']/1000:,.0f} KB"+'</td></tr>'
body+='</tbody></table><p><small>Source manifest: b1bd9b4de41b098038de1f7ff0e6629be5a40060f98ede6dc94371738d9885e1. Capture archive: 1bf95f4cbf0ac82e113b6fbeb108998742658ee19d6004d1cff4d487d68fa028. Capture duration: 149.1575 seconds including validation; no latency-baseline remeasurement.</small></p>'
current=pages/'S9-schema/current-thirtyfourth.html';assert not current.exists();current.write_text(page('S9-A candidate34 review',body))
for target in family_paths:
 family=target.parent.name[3:];old=target.read_text();archives=sorted(set(re.findall(r'href="(schema-[a-z]+/index\.html)"',old)) - {'schema-thirtyfourth/index.html'})
 body='<h1>S9-A · '+e(family)+' pages</h1><aside><strong>Actual candidate 34 · unblessed.</strong> Ordinary39 page checks passed; 447 focused tests pass. Candidate34 matrix and full gate have not run. Sparse MusicGen has a native default whose visible chip is missing; sparse Bloom exposes an alias-default count gap. Corrections are being prepared. Final preservation, approval and browser verdict remain open.</aside><p><a href="schema-thirtyfourth/index.html">Open actual model pages</a> · <a href="../S9-schema/current-thirtyfourth.html">All-family review status</a> · <a href="../../11-execution/S9-A.md">Completion sheet</a></p><h2>Earlier snapshots</h2><ul>'
 for link in archives:body+='<li><a href="'+e(link)+'">'+e(link.split('/')[0])+'</a></li>'
 body+='</ul>';target.write_text(page('S9-A '+family+' review',body))
target=pages/'S9-schema/index.html';s=target.read_text();link='<p><a href="current-thirtyfourth.html">Actual S9-A candidate34: 39 pages, measured status and sparse-input gaps</a></p>'
assert 'current-thirtyfourth.html' not in s
m=re.search(r'<body\b[^>]*>',s);s=s[:m.end()]+link+s[m.end():] if m else s+link;target.write_text(s)
checked=[]
for p in [current,*family_paths]:
 for href in re.findall(r'href="([^"]+)"',p.read_text()):
  if ':' not in href:
   assert (p.parent/href.split('#')[0]).exists(),(p,href)
   checked.append({'page':str(p.relative_to(root)),'href':href})
shutil.copyfile('/private/tmp/unfold-s9a-review34-page.py',out/'generator.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'scope':'Publication of saved actual34 pages only; no model imports or output blessing','validated_supplied_proofs':proofs,'links_checked':len(checked),'outputs':[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in [current,*targets]],'before':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted((out/'before').rglob('*')) if p.is_file()],'generator_sha256':sha(out/'generator.py')}
(out/'links.json').write_text(json.dumps(checked,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'proofs':proofs,'links_checked':len(checked),'manifest_sha256':sha(out/'manifest.json')}))
