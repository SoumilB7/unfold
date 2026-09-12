from pathlib import Path
import json,gzip,hashlib,re,subprocess
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');D=R.parent/'z-docs';O=Path(__file__).resolve().parent;P=R/'verification/receipts/S8-final-c38b008/sdxl'
def load(p):
 if p.exists():return json.loads(p.read_bytes())
 return json.loads(gzip.decompress(Path(str(p)+'.gz').read_bytes()))
def raw(p):return p.read_bytes()if p.exists()else gzip.decompress(Path(str(p)+'.gz').read_bytes())
report=load(P/'demonstration-report.json');assert report['blocking_checks']==0 and not report['blessed'];conditions={n:x['status']for n,x in report['conditions'].items()};assert all(v=='PASS'for v in conditions.values())
sparse=load(P/'sparse/controls.json');assert len(sparse)==33
hashes={}
for n in ('ordinary','sparse','misleading','rewrite','unchanged','changed','missing'):
 data=raw(P/n/'page.html');result=load(P/n/'result.json');assert hashlib.sha256(data).hexdigest()==result['html_sha256'];hashes[n]=result['html_sha256']
assert hashes['ordinary']==hashes['rewrite']==hashes['unchanged']
scratch={}
for n in ('rewrite','unchanged','changed','missing'):
 checks=report['conditions'][n]['checks'];c=next(c for c in checks if c['check']=='static reader and builder share exact scratch root bytes');values=c['evidence'];assert len(set(values.values()))==1
 actual=hashlib.sha256((P/n/'scratch-modeling-source.py').read_bytes()).hexdigest();assert actual==values['scratch_source_sha256'];scratch[n]=actual
assert '+                transformer_layers_per_block=transformer_layers_per_block[i] + 1,'in (P/'changed/source.diff').read_text()
ordinary=load(P/'ordinary/result.json');changed=load(P/'changed/result.json');assert changed['parameters']-ordinary['parameters']==89521920 and changed['constructed']-ordinary['constructed']==96
page=raw(P/'ordinary/page.html').decode();stats=re.findall(r'<div class="uf-stat-key">([^<]+)</div><div class="uf-stat-val">([^<]+)</div>',page);assert stats==[('STAGES','7'),('WEIGHTED MODULES','1,050'),('DENOISER PARAMS','2.57B')]
family={}
for n in ('sdxl','sd-v1-4'):
 x=load(R/'verification/receipts/S8-family-final-96d0c1e'/n/'summary.json');family[n]={'constructed':x['constructed'],'execution':x['execution'],'unknown_class_counts':dict(__import__('collections').Counter(y['reason_class']for y in x['family_execution_unresolved']))};assert all(y['reason_class']=='investigation_missing'for y in x['family_execution_unresolved'])
owner={}
for n in ('sdxl','sd14'):
 x=load(R/'verification/receipts'/f'S8-{n}-owner-disposition/summary.json');owner[n]={'unexplained':x.get('unexplained'), 'owner_dispositions':x['owner_dispositions'],'blessed':x['blessed']};assert not x['blessed'] and x['unexplained']==0 and sum(x['owner_dispositions'].values())==x['rows']
registered=[]
for root in ('tests/sable_test_corpus','tests/unseen_model_configs'):
 for p in (R/root).glob('*.json'):
  x=load(p);c=x.get('config',x)
  if 'UNet'in c.get('_class_name',''):registered.append(str(p.relative_to(R)))
assert sorted(registered)==['tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json','tests/unseen_model_configs/sd-v1-4.json']
changed_files=subprocess.check_output(['git','diff','--name-only','c38b008','96d0c1e','--','model_unfolder','physics'],cwd=R,text=True).splitlines();assert changed_files==['model_unfolder/renderers/html/views_diffusion.py']
oldcov=load(R/'coverage.json');previous={x['input']:{k:x[k]for k in ('proven','flagged','silent')}for x in oldcov['models']if x['input']in registered}
sheet=(D/'11-execution/S8.md').read_bytes();result={'reviewed_production':'96d0c1e (c38 source-producer campaign)','reviewed_sheet_sha256':hashlib.sha256(sheet).hexdigest(),'conditions':conditions,'sparse_equal_declared_defaults_omitted':len(sparse),'actual_html_sha256':hashes,'actual_scratch_source_sha256':scratch,'changed_parameter_delta':89521920,'changed_occurrence_delta':96,'actual_banner':stats,'family_accounting':family,'owner_dispositions':owner,'registered_unet_inputs':registered,'production_delta_after_c38':changed_files,'prior_coverage_not_final':previous,'scope':'C8 artifact completeness and sheet claims only; no models/tests/production edits; final matrix and broad gate pending outside this review.'};(O/'checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');(O/'reviewed-S8.md').write_bytes(sheet);print(json.dumps({'conditions':conditions,'sparse_omissions':len(sparse),'actual_banner':stats,'registered_unet_inputs':registered,'source_and_html_checks':'PASS'},indent=2))
