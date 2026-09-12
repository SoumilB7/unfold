from pathlib import Path
import gzip,hashlib,io,json,tarfile
base=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S9-A-schema-candidate-fff99e22')
out=base/'ladder-review-supplement';out.mkdir(exist_ok=False)
names=['unfold-s9a-class-default-value-preparation','unfold-s9a-class-default-value-preparation-v2','unfold-s9a-class-default-value-preparation-v3','unfold-s9a-class-default-value-preparation-v4','unfold-s9a-class-default-value-preparation-v5','unfold-s9a-independent-default-d1-v2-review','unfold-s9a-independent-default-d1-v3-review','unfold-s9a-independent-default-d1-v4-review','unfold-s9a-independent-five-declarations-review','unfold-s9a-independent-operand-correction-review','unfold-s9a-independent-operand-root-review','unfold-s9a-operand-root-snapshot','unfold-s9a-sparse-eighteenth-llama','unfold-s9a-sparse-eighteenth-llama-lane']
files={}
for name in names:
 root=Path('/private/tmp')/name
 assert root.is_dir(),root
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),p
  if p.is_file():files[name+'/'+str(p.relative_to(root))]=(p,p.read_bytes())
archive=out/'artifacts.tar.gz'
with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
 for name,(p,data) in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
with tarfile.open(archive,'r:gz') as tar:
 assert tar.getnames()==list(files)
 for name,(p,data) in files.items():assert tar.extractfile(name).read()==data==p.read_bytes(),name
manifest={'status':'IN_PROGRESS_NOT_APPROVED','scope':'Completed source review snapshots plus actual eighteenth sparse failure; no green relabel','files':[{'member':name,'original':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,(p,data) in files.items()],'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'every_member_restored_and_compared':True,'product_imports_or_tests_run':False}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'archive.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'files':len(files),'archive_bytes':archive.stat().st_size,'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()},indent=2))
