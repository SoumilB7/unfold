"""Report-only adversarial sidecar checks; no model or pytest execution.

Input is a previously generated ordinary page. Only temporary copies of
review artifacts are modified, never production files or blessed outputs.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT=Path.cwd(); sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from report_s8_demonstration import read_case, condition_checks, claim_traces, blocks, observation
source=Path('/private/tmp/unfold-s8-d3a4d61-demonstration/ordinary')
results={}
with tempfile.TemporaryDirectory(prefix='s8-report-review-') as directory:
 root=Path(directory); target=root/'ordinary';target.mkdir()
 for name in ('result','facts','ir','controls','observation','qualified-facts'):
  shutil.copyfile(source/(name+'.json'),target/(name+'.json'))
 (target/'cited-source').symlink_to(source/'cited-source',target_is_directory=True)
 ordinary=read_case(target)
 checks=condition_checks(ordinary,ordinary)
 results['missing_actual_page']={'actual_page_exists':(target/'page.html').exists(),'checks':checks,
                                 'false_positive':next(row for row in checks if row['check']=='actual HTML generated')['status']=='PASS'}
 baseline_traces=claim_traces(root)
 ir=json.loads((target/'ir.json').read_text())
 for block in blocks(ir.get('extras',{}).get('render',{})):
  if block.get('source_instance_path'): block['source_fact_keys']=[]
 (target/'ir.json').write_text(json.dumps(ir))
 disconnected=claim_traces(root)
 results['removed_block_fact_links']=[{'claim':after['claim'],'original_chain_gaps':before.get('chain_gaps'),
  'changed_chain_gaps':after.get('chain_gaps'),'block_fact_keys':after.get('block',{}).get('source_fact_keys'),
  'canonical_fact':after.get('canonical_fact',{}).get('key'),'status':after['status']}
  for before,after in zip(baseline_traces,disconnected)]
 forged=copy.deepcopy(ordinary)
 forged['result']['condition']='unchanged';forged['result']['same_source']=True
 forged['result']['static_source_sha256']='a'*64
 forged['result']['runtime_source_sha256']='b'*64
 forged['result']['scratch_source_sha256']='c'*64
 checks=condition_checks(ordinary,forged)
 results['mismatched_scratch_hashes']=[row for row in checks if row['check'] in {'static reader and builder share exact scratch root bytes','unchanged scratch is byte-identical HTML'}]
page=(source/'page.html').read_text()
facts_a=json.loads((source/'facts.json').read_text());facts_b=copy.deepcopy(facts_a)
new_key='root.denoiser.primary_state_routes'
facts_a[new_key]={'value':{'source_port':'first'}}
facts_b[new_key]={'value':{'source_port':'different'}}
ir=json.loads((source/'ir.json').read_text())
obs_a=observation(ir,facts_a,page);obs_b=observation(ir,facts_b,page)
results['new_projected_fact_omitted_from_semantics']={'fact_key':new_key,'raw_values_differ':facts_a[new_key]!=facts_b[new_key], 'semantic_facts_equal':obs_a['semantic_facts']==obs_b['semantic_facts'], 'selected_fact_keys':sorted(obs_a['semantic_facts']), 'scope':'hypothetical new cutover fact; existing fixed selection cannot cover it without update'}
print(json.dumps({'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'saved_ordinary':str(source),'saved_result_sha256':hashlib.sha256((source/'result.json').read_bytes()).hexdigest(),
 'scope':'report integrity and linkage only; generator and semantic acceptance not bypassed by this probe',
 'results':results},indent=2,sort_keys=True))
