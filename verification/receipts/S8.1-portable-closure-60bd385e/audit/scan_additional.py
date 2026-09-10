"""Scan explicit new/changed files only; never rescan historical archive tree."""
from pathlib import Path
import argparse,ast,collections,gzip,hashlib,io,json,re,tarfile,zipfile
parser=argparse.ArgumentParser();parser.add_argument('--paths-json',required=True);parser.add_argument('--output',required=True);parser.add_argument('--excluded-self-reports-json');args=parser.parse_args()
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
source=Path('/private/tmp/unfold-s81-approved-portability-audit/scan.py')
raw_source=source.read_bytes();tree=ast.parse(raw_source);names={'HOST','REF','BAD','KEYS'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'digest','category','analyze','walk'} or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
ns=globals().copy();ns.update(cache={},rows=[],containers=collections.Counter());exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned base scanner definitions>','exec'),ns)
base_category=ns['category'];ns['category']=lambda p:'production_source' if p.startswith(('model_unfolder/','physics/')) else base_category(p)
paths=json.loads(Path(args.paths_json).read_text());assert isinstance(paths,list) and all(isinstance(p,str) for p in paths) and len(paths)==len(set(paths))
excluded=json.loads(Path(args.excluded_self_reports_json).read_text()) if args.excluded_self_reports_json else []
assert set(paths).isdisjoint(excluded)
files=[];errors=[]
for p in paths:
 try:
  rel=Path(p);assert not rel.is_absolute() and '..' not in rel.parts
  f=root/rel;raw=f.read_bytes();files.append({'path':p,'sha256':ns['digest'](raw),'bytes':len(raw),'symlink':f.is_symlink()});ns['walk'](raw,p)
 except Exception as e:errors.append({'path':p,'type':type(e).__name__,'error':str(e)})
# Include archive-link and tracked-symlink target bytes; never follow archive links.
for row in ns['rows']:
 if row['format']=='archive_link':row.update(ns['analyze'](row['target'].encode()),format='archive_link')
for row in files:
 if row['symlink']:
  import os
  target=os.readlink(root/row['path']);ns['rows'].append({'logical':row['path']+'!symlink-target','category':ns['category'](row['path']),**ns['analyze'](target.encode()),'format':'symlink_target'})
out=Path(args.output);out.mkdir(exist_ok=False)
(out/'files.json').write_text(json.dumps(files,indent=2)+'\n')
with (out/'findings.jsonl').open('w') as stream:
 for row in ns['rows']:stream.write(json.dumps(row,separators=(',',':'))+'\n')
summary={'status':'COMPLETE' if not errors else 'INCOMPLETE','base_scanner_sha256':hashlib.sha256(raw_source).hexdigest(),'supplement_scanner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_paths_sha256':hashlib.sha256(Path(args.paths_json).read_bytes()).hexdigest(),'files':len(files),'stored_bytes':sum(r['bytes'] for r in files),'decoded_leaves':len(ns['rows']),'containers':dict(ns['containers']),'host_path_files':sum(bool(r.get('host_path_token_occurrences')) for r in ns['rows']),'host_path_tokens':sum(r.get('host_path_token_occurrences',0) for r in ns['rows']),'known_local_digest_files':sum(bool(r.get('known_local_index_digest_occurrences')) for r in ns['rows']),'errors':errors,'excluded_self_descriptive_audit_reports':excluded,'scope':'Explicit changed/new captured files only; old archive contents not rescanned. Self-descriptive audit copies excluded explicitly; they are reports of prior findings, not active portable authority. Counts are separate snapshot supplement, not blindly added to baseline counts for duplicate changed paths.'}
(out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert not errors
