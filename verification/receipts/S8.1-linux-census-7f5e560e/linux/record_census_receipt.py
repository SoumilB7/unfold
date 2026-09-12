from pathlib import Path
import hashlib,json,shutil
base=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer');repo=base/'unfold-pkg';out=repo/'verification/receipts/S8.1-linux-census-7f5e560e';out.mkdir(exist_ok=False)
check=Path('/private/tmp/unfold-s81-linux-census-check/result.json');r=json.loads(check.read_text());assert r['status']=='PASS_COMMITTED_CENSUS_CHECK' and r['lane']['passed']
files={}
for ns,folder in [('diagnostic','/private/tmp/unfold-s81-linux-census-diagnostic'),('independent','/private/tmp/unfold-s81-linux-census-independent-review'),('check','/private/tmp/unfold-s81-linux-census-check')]:
 for p in Path(folder).iterdir():
  if p.is_file():files[ns+'/'+p.name]=p
for name in ('push-result.json','push-failed.log','diagnose_census.py','check_census.py','record_census_receipt.py'):
 files['linux/'+name]=Path('/private/tmp/unfold-s81-approved-linux')/name
entries={}
for name,p in files.items():
 raw=p.read_bytes();dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);assert dest.read_bytes()==p.read_bytes();entries[name]={'source':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
(out/'manifest.json').write_text(json.dumps(entries,indent=2,sort_keys=True)+'\n')
(out/'README.md').write_text('''# S8.1 Linux census freshness correction\n\nThe first pushed closure `7f5e560e` failed [Linux run34452333662](https://github.com/SoumilB7/unfold/actions/runs/34452333662) at config-consumption census freshness. Native sandbox proof passed; all later gates were skipped.\n\nThe unchanged producer reproduced the complete Linux diff in a committed isolated checkout. The census document dated from S4; S8.md already recorded the SDXL accounting limitations. The new document preserves two UNCLASSIFIED reads (`act_fn`, `norm_eps`),32 additional pending occurrences and incomplete denoiser accounting. No new consumption claim, disposition, source or checker change was made.\n\nIndependent exact-delta approval is in `independent/actual-candidate-review.json`. Generated-document commit `8d7bcdea` changed only `docs/COR5_NET1_MIGRATION_DEBT.md`. The unchanged committed-tree `census.py --check` passed, with matching source/blessed-artifact fingerprints; see `check/result.json`. The original Linux failure and complete candidate/diff are retained.\n\nThis is a generated discovery-report refresh, not S9 migration or a product/preservation re-bless. The full Linux workflow must still pass at the next pushed head. Original S8.1 local output/preservation/budget checks remain in [the closure receipt](../S8.1-approved-closure-b3624f2a/README.md).\n''')
print('Retained',len(entries),'exact census correction receipt files')
