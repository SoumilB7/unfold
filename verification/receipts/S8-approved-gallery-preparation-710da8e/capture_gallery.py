"""Render pinned actual HTML with the unchanged project gallery API; never parse."""
from pathlib import Path
import argparse,copy,dataclasses,hashlib,json,shutil,subprocess,sys,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(v):
 if dataclasses.is_dataclass(v):return canon(dataclasses.asdict(v))
 if isinstance(v,dict):return {k:canon(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [canon(x) for x in v]
 return v
def write(p,v):p.write_text(json.dumps(canon(v),indent=2,sort_keys=True)+'\n')
def manifest(root):return {str(p.relative_to(root)):sha(p) for d in ('model_unfolder','physics','scripts') for p in sorted((root/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--slug',required=True);a=p.parse_args();mp=Path(a.manifest);m=json.loads(mp.read_text());row=next(x for x in m['cases'] if x['slug']==a.slug);root=Path(m['checkout']);out=Path(m['output_root'])/a.slug;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();before=manifest(root);pins={str(mp):sha(mp),str(Path(__file__).resolve()):sha(__file__),**row['input_sha256'],m['rsvg_binary']:m['rsvg_binary_sha256']};write(out/'source-before.json',before);write(out/'input-pins.json',pins)
 try:
  assert before==m['production_source_sha256'];assert sha(__file__)==m['script_sha256'];assert all(sha(path)==h for path,h in row['input_sha256'].items());assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==m['production_commit']
  assert shutil.which('rsvg-convert')==m['rsvg_binary'] and sha(m['rsvg_binary'])==m['rsvg_binary_sha256']
  sys.path.insert(0,str(root));from model_unfolder import preview
  from model_unfolder.sable import SableReport,SableCheck
  assert Path(preview.__file__).resolve().is_relative_to(root)
  original=json.loads(Path(row['report']).read_text());fields=copy.deepcopy(original);fields['checks']=[SableCheck(**x) for x in fields['checks']];report=SableReport(**fields);assert canon(report)==original
  htmlbytes=Path(row['html']).read_bytes();html=htmlbytes.decode();views=preview.svg_views(html);distinct=[];seen=set()
  for label,svg in views:
   key=preview._visual_hash(svg)
   if key in seen:continue
   seen.add(key);distinct.append({'label':label,'visual_hash':key,'svg_sha256':hashlib.sha256(svg.encode()).hexdigest(),'svg':svg})
  signature=[[x['label'],x['visual_hash']] for x in distinct];old_signature=canon(report.view_hashes);changes=[]
  if a.slug=='bloom':
   assert row['report_phase']=='actual736-report-plus-approved-final-render';assert len(signature)==len(old_signature)==4
   changes=[{'index':i,'old':x,'final':y} for i,(x,y) in enumerate(zip(old_signature,signature)) if x!=y]
   assert len(changes)==1 and changes[0]['index']==1 and changes[0]['old'][0]==changes[0]['final'][0]=='attn'
   assert changes==row['approved_bloom_signature_delta'];report.view_hashes=[tuple(x) for x in signature]
  else:assert signature==old_signature,'Actual HTML does not match retained Sable view signature'
  assert len(distinct)==row['distinct_views'];assert report.mechanical_passed and report.oracle=='present'
  class SavedHtmlCapture:
   def to_html(self,standalone=True):
    assert standalone is True
    return html
  gallery=preview.render_images(SavedHtmlCapture(),str(out),**m['render_options']);assert len(gallery)==len(distinct)
  image_rows=[]
  for image,v in zip(gallery,distinct):
   image_rows.append({'file':Path(image).name,'png_sha256':sha(image),'label':v['label'],'visual_hash':v['visual_hash'],'source_svg_sha256':v['svg_sha256'],'inspected':False})
  report.gallery=gallery;semantic=canon(report);semantic['gallery']=original['gallery']
  if a.slug=='bloom':semantic['view_hashes']=original['view_hashes']
  assert semantic==original,'Report semantic fields changed'
  assert Path(row['html']).read_bytes()==htmlbytes
  manifest_names=[line.split()[0] for line in (out/'MANIFEST.txt').read_text().splitlines() if line and not line.startswith('#')];assert manifest_names==[x['file'] for x in image_rows]
  write(out/'report.json',report);write(out/'image-bindings.json',image_rows);write(out/'signature-delta.json',changes)
  write(out/'result.json',{'status':'GENERATED_PENDING_INDIVIDUAL_PIXEL_REVIEW','slug':a.slug,'model':report.model,'images':len(gallery),'actual_html_sha256':sha(row['html']),'original_report_sha256':sha(row['report']),'report_sha256':sha(out/'report.json'),'manifest_sha256':sha(out/'MANIFEST.txt'),'render_options':m['render_options'],'rsvg_binary':shutil.which('rsvg-convert'),'rsvg_version':subprocess.check_output(['rsvg-convert','--version'],text=True).strip(),'source_phase':row['report_phase'],'new_sable_execution':False,'blessed':False,'elapsed_seconds':round(time.monotonic()-start,3)})
  print(a.slug,len(gallery),'PNGs generated; individual inspection pending',flush=True)
 except BaseException as e:
  write(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc()});raise
 finally:
  after=manifest(root);actual={path:sha(path) for path in pins};write(out/'source-after.json',after);write(out/'pin-check.json',{'source_equal':before==after,'input_script_equal':actual==pins});assert before==after and actual==pins
if __name__=='__main__':main()
