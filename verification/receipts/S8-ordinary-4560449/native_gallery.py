import argparse, hashlib, html, json, shutil, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--page',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
sys.path.insert(0,str(a.checkout))
from model_unfolder.preview import render_images,svg_views,_visual_hash
raw=a.page.read_bytes();page=raw.decode();a.output.mkdir(parents=True,exist_ok=True)
class Saved:
 def to_html(self,**kwargs):return page
images=render_images(Saved(),str(a.output),scale=1,highlight_clickable=True);seen=set();rows=[]
for label,svg in svg_views(page):
 key=_visual_hash(svg)
 if key in seen:continue
 seen.add(key);png=Path(images[len(rows)]);svgpath=png.with_suffix('.svg');svgpath.write_text(svg);rows.append({'label':label,'png':png.name,'svg':svgpath.name,'visual_hash':key})
m={'scope':'Intermediate fresh production output; not final S8 acceptance','commit':a.commit,'html_sha256':hashlib.sha256(raw).hexdigest(),'source_page':str(a.page),'baked_svgs':len(svg_views(page)),'distinct_views':len(rows),'views':rows,'pixel_inspection':'pending'}
(a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S8 '+a.commit+' intermediate gallery</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:24px}img{max-width:100%;border:1px solid #ccc}figure{margin:35px 0}.notice{padding:20px;background:#fff2db}</style><h1>S8 '+a.commit+' — intermediate production gallery</h1><p class="notice">Actual fresh production HTML rendered at the immutable checkpoint shown. Native SVG images do not establish document-level browser behavior. No final S8 acceptance or blessing. Amber outlines mark clickable nodes in this image-only overlay.</p><p><a href="manifest.json">Exact manifest</a></p>'+''.join('<figure><figcaption>'+html.escape(row['label'])+'</figcaption><a href="'+row['svg']+'">SVG</a><br><img loading="lazy" src="'+row['png']+'"></figure>' for row in rows))
print(json.dumps({k:v for k,v in m.items() if k!='views'}))
