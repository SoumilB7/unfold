"""Finalize reviewable S10-1 receipt; never install the pending S7 candidate."""
import hashlib,json,shutil,subprocess
from pathlib import Path
R=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');T=R/'.claude/worktrees/verify-s10-1-deletion';S=Path('/private/tmp/unfold-s10-1');O=R/'verification/receipts/S10-1-unet-deletion-fff99e22';sha=lambda b:hashlib.sha256(b).hexdigest()
owned=json.loads((S/'owned-files.json').read_bytes())+['docs/U3_CURRENT_READER_INVENTORY.md','test_support/latency_contract.py','tests/test_s81_latency.py','verification/latency_budgets.json']
assert len(owned)==len(set(owned))
for rel in owned:
 assert (R/rel).exists()==(T/rel).exists(),rel
 if (R/rel).exists():assert (R/rel).read_bytes()==(T/rel).read_bytes(),rel
C=O/'closure';C.mkdir(exist_ok=True);assert not (C/'summary.json').exists()
for src in [S/'baseline-renewal.json',S/'qualification-latency-ratchet/result.json',S/'qualification-latency-ratchet/latency-contract.log',S/'qualification-latency-ratchet/verify.py']:
 shutil.copyfile(src,C/('latency-'+src.name if src.parent.name=='qualification-latency-ratchet' else src.name))
shutil.copyfile(Path(__file__),C/'finish_receipt.py')
(C/'owned-files.json').write_text(json.dumps(owned,indent=2)+'\n')
patch=subprocess.check_output(['git','diff','--binary','fff99e22','--',*owned],cwd=R);(C/'final-s10-only.patch').write_bytes(patch)
raw=subprocess.check_output(['git','diff','--numstat','fff99e22','--',*owned],cwd=R,text=True)
nums=[line.split('\t',2) for line in raw.splitlines()];prod=[v for v in nums if v[2].startswith('model_unfolder/')]
summary={'status':'IMPLEMENTED_SCOPED_CHECKS_PASS_PENDING_S7_STAMP_APPROVAL_AND_BROAD_GATE','base_commit':'fff99e2230064762598748463accd170e1e34e22','source_patch_sha256':sha(patch),'owned_files':owned,'growth':{'production_added':sum(int(v[0]) for v in prod),'production_deleted':sum(int(v[1]) for v in prod),'tracked_unit_added':sum(int(v[0]) for v in nums),'tracked_unit_deleted':sum(int(v[1]) for v in nums),'legacy_modules_deleted':2,'legacy_renderers_deleted':5,'quarantined_readers_deleted':5,'raw_parse_sites_deleted':1,'extras_debts_deleted':3,'renderer_debts_deleted':4,'broad_excepts_deleted':4,'temporary_bridges_added':0,'temporary_bridges_retired':1},'parity_witnesses':8,'parity_surface_comparisons':64,'product_deltas':0,'checks':{'focused':84,'authority':{'original_passed':43,'total':44,'corrected_exception_file_passed':4},'latency_contract':45,'latency_processes':12,'fallback_grep_hits':0},'s7_stamp_installed':False,'s7_stamp_candidate_sha256':'a08d366d018626ab87b69230db977c46a117634aa98bbd8392830bf586d1d490','commit_created':False,'pushed':False}
# Eight retained raw surfaces per witness, including config and typed failures.
assert all(len(json.loads((S/'parity/before'/row['witness']/'manifest.json').read_bytes()))==8 for row in json.loads((S/'parity/result.json').read_bytes())['rows'])
(C/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={str(p.relative_to(O)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(O.rglob('*')) if p.is_file()}
(O/'closure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
