import gzip,hashlib,json,shutil,re
from pathlib import Path
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
src=Path('/private/tmp/unfold-s8-coverage-7168440')
out=root/'verification/receipts/S8-final-coverage-7168440'
for phase in ('generate','check'):
 d=json.loads((src/f'{phase}-receipt.json').read_text());assert d['exit_code']==0 and d['implementation_unchanged']
assert json.loads((src/'implementation-before.json').read_text())==json.loads((src/'implementation-after.json').read_text())
assert (root/'coverage.json').read_bytes()==Path('/private/tmp/unfold-s8-final-verification-7168440/coverage.json').read_bytes()
comparisons=[]
for name in ('stable-diffusion-xl-base-1-0','sd-v1-4'):
 a=src/'actual-generate'/name;b=src/'actual-check'/name
 for item in ('ir.json.gz','display-ir.json.gz'):
  assert gzip.decompress((a/item).read_bytes())==gzip.decompress((b/item).read_bytes()), (name,item)
 ah=gzip.decompress((a/'page.html.gz').read_bytes()).decode();bh=gzip.decompress((b/'page.html.gz').read_bytes()).decode()
 am=set(re.findall(r'uf-[a-z0-9]{10}',ah));bm=set(re.findall(r'uf-[a-z0-9]{10}',bh));assert len(am)==len(bm)==1
 assert ah.replace(next(iter(am)),next(iter(bm)))==bh
 comparisons.append({'model':name,'actual_ir_and_display_ir_bytes_identical':True,'actual_html_only_exact_mount_token_differs':True,'generate_html_sha256':hashlib.sha256(ah.encode()).hexdigest(),'check_html_sha256':hashlib.sha256(bh.encode()).hexdigest(),'mounts':[next(iter(am)),next(iter(bm))],'diagnostic_only_original_pages_untouched':True})
(src/'generate-check-output-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
out.mkdir(exist_ok=True);rows=[]
for p in sorted(src.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(src);data=p.read_bytes();target=out/rel
 if len(data)>500000 and p.suffix!='.gz':target=Path(str(target)+'.gz');encoded=gzip.compress(data,mtime=0)
 else:encoded=data
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(encoded)
 rows.append({'original':str(rel),'artifact':str(target.relative_to(out)),'original_sha256':hashlib.sha256(data).hexdigest(),'artifact_sha256':hashlib.sha256(encoded).hexdigest()})
for name in ('unfold-s8-final-coverage-runner-7168440.py','unfold-s8-coverage-capture.py','unfold-s8-persist-coverage-7168440.py'):
 p=Path('/private/tmp')/name;shutil.copy2(p,out/name)
(out/'artifact-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'README.md').write_text('''# Final actual44 coverage at7168440

PASS: both actual generation and the independent live `--check` completed the unchanged29corpus+15unseen denominator. The serial runner pins full production/script manifests and its observer helper before/after; all remain unchanged. Exact commands, timings, full logs, per-model comparisons and actual public SDXL/SD14 IR/display-IR/HTML captures from both passes are retained. This is a fresh44-input coverage campaign, separate from the historical25+1+1+12 derived39 execution matrix.

Measured totals are645proven/294visibly flagged/0silent, compared with accepted83140f1 totals621/241/0. Only SDXL(+12proven/+37flagged) and SD-v1-4(+12/+16) differ; every complete row for the other42 models is unchanged. All44non-silent fields equal the original5227red denominator; its620silent findings are resolved by the separately reviewed typed-contract/label/actual-receipt corrections. Original red outputs remain preserved. The owner comparison and public-output controls are in S8-coverage-owner-final-7168440 and S8-actual-public-output-control-7168440.

README and the exact-count release assertion are refreshed to these measured645/294 totals. The29+15denominator, silent0 requirement and all audit behavior remain unchanged. This does not bless product or preservation output baselines; the isolated broad coordinator remains separate. Every original capture is retained losslessly, with original and stored hashes in artifact-manifest.json.
''')
print(out)
