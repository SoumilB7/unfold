from pathlib import Path
import ast,difflib,hashlib,json,shutil
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg');r=root/'verification/receipts/S9-A-schema-candidate-fff99e22/new-resume'
out=r/'combined35-source-proposal';out.mkdir(exist_ok=True);assert not list(out.iterdir())
origins_config=[('layer-count-alias-default-proposal-r2','source','c4f3ab81462e669b8d26f86f99a1107046b67b3b58dc46d4d05a9afc391d8a4c'),('repeated-schedule-visibility-proposal-r2','proposed','24050ff7373e377a9a5997ab753cebae29b35b54311a33812740b445a255278e')]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
parser='model_unfolder/adapters/transformer/parser.py';base=(root/parser).read_text();bl=base.splitlines(keepends=True);edits=[];origins=[];test_sources={}
for name,sub,pin in origins_config:
 m=r/name/'manifest.json';assert h(m)==pin;origins.append({'path':str(m.relative_to(root)),'sha256':pin});data=json.loads(m.read_text())
 for row in data['files']:
  p=r/name/sub/row['path'];assert h(p)==row['proposed_sha256'];assert h(root/row['path'])==row['before_sha256']
  if row['path']!=parser:test_sources[row['path']]=p
 pl=(r/name/sub/parser).read_text().splitlines(keepends=True)
 for tag,i,j,a,b in difflib.SequenceMatcher(None,bl,pl,autojunk=False).get_opcodes():
  if tag!='equal':edits.append((i,j,pl[a:b],name))
for a,b in zip(sorted(edits),sorted(edits)[1:]):assert a[1]<=b[0],(a,b)
combined=list(bl)
for i,j,lines,name in sorted(edits,reverse=True):combined[i:j]=lines
p=out/'proposed'/parser;p.parent.mkdir(parents=True);p.write_text(''.join(combined));ast.parse(p.read_text());files=[parser]
for rel,src in test_sources.items():
 p=out/'proposed'/rel;p.parent.mkdir(exist_ok=True);shutil.copyfile(src,p);ast.parse(p.read_text());files.append(rel)
rows=[{'path':p,'before_sha256':h(root/p),'proposed_sha256':h(out/'proposed'/p)} for p in files]
(out/'changes.patch').write_text(''.join(''.join(difflib.unified_diff((root/p).read_text().splitlines(keepends=True) if (root/p).is_file() else [],(out/'proposed'/p).read_text().splitlines(keepends=True),fromfile='a/'+p,tofile='b/'+p)) for p in files))
shutil.copyfile(__file__,out/'compose.py')
manifest={'scope':'Exact disjoint source composition of two immutable author proposals; pending independent verdict; no main source edits or runtime.','origins':origins,'files':rows,'parser_edit_intervals':[{'start':i,'end':j,'author':n} for i,j,lines,n in edits],'composition_script_sha256':h(out/'compose.py')}
p=out/'manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n');print(h(p));print(json.dumps(rows,indent=2))
