"""Read-only tracked artifact census. Decode containers, never execute payloads."""
from pathlib import Path
import ast,collections,gzip,hashlib,io,json,re,subprocess,tarfile,zipfile,time
ROOT=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
OUT=Path('/private/tmp/unfold-s81-approved-portability-audit')
HOST=re.compile(rb'(?<![A-Za-z0-9_])(?:/(?:Users|Library|private|tmp|var|opt|home|root|usr|Applications|workspace|workspaces|mnt|runner|app|code|mac|linux|src|fixture|fake|path)(?:/[^\s\x00\x01\x02<>"\'`\\,;\]\[{}()]*)?|[A-Za-z]:[\\/][^\s\x00<>"\'`]*|file://[^\s<>"\'`]*)')
REF=re.compile(rb'(?:ProgramIndex|(?:self\.)?(?:graph|bindings)\.index|self\.index|index)\.fingerprint\b')
BAD=[x.encode() for x in ['09fc7b6a95f66c670862b0660094bce1078a3b341e2c963b5da40d3e19673f9d','14f56c8ae9acd61f6cf43b673a4143dcb3593dfa2124757f63bdcc635d5a637e','e27e0a82b761ed6805a5be30fe0b0a6e57f119cdcfb409916e797e0b98a18824','f3a8764adaa00599fb7c6e56538805d3860902f9d4a3c387b2721d2b51336553']]
KEYS={'index_fingerprints','index_fingerprint','source_index_fingerprint','canonical_path'}
cache={}; rows=[]; errors=[];containers=collections.Counter(); started=time.time()
def digest(b):return hashlib.sha256(b).hexdigest()
def category(p):
 if p.startswith('verification/s7/'):return 'active_s7'
 if p.startswith('verification/s6/'):return 'legacy_s6'
 if p.startswith('verification/receipts/'):return 'retained_receipts'
 if p.startswith('examples/'):return 'active_examples'
 if p.startswith('tests/') and not p.endswith('.py'):return 'test_fixture'
 if p.startswith('tests/'):return 'test_source'
 return 'verification_metadata'
def analyze(raw):
 h=digest(raw)
 if h in cache:return cache[h]
 hosts=list(HOST.finditer(raw));refs=list(REF.finditer(raw));known={x.decode():raw.count(x) for x in BAD if x in raw}
 result={'decoded_sha256':h,'bytes':len(raw),'host_path_token_occurrences':len(hosts),'host_path_examples':list(dict.fromkeys(x.group().decode('utf8','replace') for x in hosts))[:5],'local_fingerprint_source_mentions':len(refs),'known_local_index_digest_occurrences':known,'index_fields':[],'host_json_leaf_count':0,'host_json_examples':[]}
 try: text=raw.decode('utf8')
 except UnicodeDecodeError:result['format']='opaque_binary';cache[h]=result;return result
 result['format']='text'
 if text.lstrip().startswith(('{','[')):
  try:value=json.loads(text)
  except (ValueError,RecursionError):result['format']='non_json_text'
  else:
   result['format']='json'; stack=[('$',value)]
   while stack:
    path,v=stack.pop()
    if isinstance(v,dict):
     for k,c in v.items():
      kp=path+'.'+k
      if k in KEYS and k!='canonical_path':result['index_fields'].append({'path':kp,'value':c})
      stack.append((kp,c))
    elif isinstance(v,list):stack.extend((path+f'[{i}]',c) for i,c in enumerate(v))
    elif isinstance(v,str) and HOST.search(v.encode('utf8','surrogatepass')):
     result['host_json_leaf_count']+=1
     if len(result['host_json_examples'])<5:result['host_json_examples'].append({'path':path,'value':v[:250]})
 cache[h]=result;return result
def walk(raw,logical,depth=0):
 if depth>8:raise ValueError('container nesting >8')
 if raw[:2]==b'\x1f\x8b':
  containers['gzip']+=1;return walk(gzip.decompress(raw),logical+'!gzip',depth+1)
 if len(raw)>262 and raw[257:262]==b'ustar':
  containers['tar']+=1
  with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as t:
   for member in t:
    if member.isfile():walk(t.extractfile(member).read(),logical+'!tar/'+member.name,depth+1)
    elif not member.isdir():rows.append({'logical':logical+'!tar/'+member.name,'category':category(logical.split('!')[0]),'format':'archive_link','target':member.linkname})
  return
 if raw[:4]==b'PK\x03\x04':
  containers['zip']+=1
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   for member in z.infolist():
    if not member.is_dir():walk(z.read(member),logical+'!zip/'+member.filename,depth+1)
  return
 rows.append({'logical':logical,'category':category(logical.split('!')[0]),**analyze(raw)})
paths=[p for p in subprocess.check_output(['git','ls-files','-z','verification','examples','tests'],cwd=ROOT).decode().split('\0') if p]
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();files=[]
for i,p in enumerate(paths):
 try:
  f=ROOT/p;raw=f.read_bytes();files.append({'path':p,'sha256':digest(raw),'bytes':len(raw),'symlink':f.is_symlink()});walk(raw,p)
 except Exception as e:errors.append({'path':p,'type':type(e).__name__,'error':str(e)})
 if i%500==0:print('progress',i,'of',len(paths),'seconds',round(time.time()-started,1),flush=True)
(OUT/'files.json').write_text(json.dumps(files,separators=(',',':'))+'\n')
with (OUT/'findings.jsonl').open('w') as stream:
 for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
summary={}
for cat in sorted(set(r['category'] for r in rows)):
 rs=[r for r in rows if r['category']==cat];summary[cat]={'decoded_leaf_count':len(rs),'decoded_bytes':sum(r.get('bytes',0) for r in rs),'host_path_files':sum(bool(r.get('host_path_token_occurrences')) for r in rs),'host_path_token_occurrences':sum(r.get('host_path_token_occurrences',0) for r in rs),'host_json_leaves':sum(r.get('host_json_leaf_count',0) for r in rs),'known_local_digest_files':sum(bool(r.get('known_local_index_digest_occurrences')) for r in rs),'index_field_count':sum(len(r.get('index_fields',[])) for r in rs),'local_fingerprint_source_mention_files':sum(bool(r.get('local_fingerprint_source_mentions')) for r in rs)}
result={'status':'COMPLETE' if not errors else 'INCOMPLETE','start_head':head,'end_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'tracked_file_count':len(paths),'read_file_count':len(files),'stored_bytes':sum(r['bytes'] for r in files),'containers':dict(containers),'unique_decoded_leaf_contents':len(cache),'decoded_leaf_count':len(rows),'categories':summary,'errors':errors,'seconds':time.time()-started,'limitations':['Host-path detection is a declared token search, not a proof that arbitrary opaque hashes are portable.','Source mentions and known historical digests are separate from active typed proof seals.','Opaque binary streams inspected as bytes; no pickle/pstats execution or deserialization.','Any concurrent install must be reconciled by recorded file hashes and final active scan.']}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
