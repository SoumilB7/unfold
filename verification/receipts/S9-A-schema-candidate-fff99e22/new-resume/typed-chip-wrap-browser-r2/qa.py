"""Saved-artifact browser QA. Requires isolated playwright venv; imports no product code."""
from pathlib import Path
import json,hashlib,time,platform,importlib.metadata,re
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;ROOT=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer');PAGES=HERE/'inputs'
OUT=HERE/'results';OUT.mkdir(exist_ok=False)
CASES=[('llama','S9-transformer/schema-thirtysixth/llama-7b.html','attn',False),('sdxl','S9-unet/schema-thirtysixth/stable-diffusion-xl-base-1-0.html','denoiser',True),('bloom-ordinary','S9-schema/defaults-thirtysixth/bloom/ordinary.html','attn',False),('bloom-sparse','S9-schema/defaults-thirtysixth/bloom/sparse.html','attn',True),('musicgen-ordinary','S9-schema/defaults-thirtysixth/musicgen/ordinary.html','cross_attn',False),('musicgen-sparse','S9-schema/defaults-thirtysixth/musicgen/sparse.html','cross_attn',True)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
pins={str(PAGES/path):sha(PAGES/path) for _,path,_,_ in CASES};findings=[];results={}
COUNT_JS="""() => ({elements:document.querySelectorAll('*').length,cards:document.querySelectorAll('[data-card-id]').length,op_nodes:document.querySelectorAll('.uf-node').length,visible_cards:[...document.querySelectorAll('.uf-card-detail')].filter(n=>n.getBoundingClientRect().width>0).map(n=>n.dataset.cardId),chip_kinds:[...document.querySelectorAll('[data-chip-kind]')].map(n=>n.dataset.chipKind),viewport:{width:innerWidth,height:innerHeight},document:{width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight}})"""
with sync_playwright() as pw:
 browser=pw.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
 versions={'python':platform.python_version(),'platform':platform.platform(),'chrome':browser.version,'playwright':importlib.metadata.version('playwright'),'executable':'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','profile':'Playwright-created fresh temporary profile; no personal profile accessed','network':'HTTP(S) blocked for saved-artifact scope; external Google Fonts use CSS fallback','viewport':{'width':1440,'height':1000}}
 for name,rel,card_id,expect_default in CASES:
  folder=OUT/name;folder.mkdir();ctx=browser.new_context(viewport=versions['viewport'],device_scale_factor=1);page=ctx.new_page();errors=[];console=[];blocked=[]
  page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:console.append({'type':m.type,'text':m.text}) if m.type in ['warning','error'] else None)
  def route(r):
   if r.request.url.startswith(('http://','https://')):blocked.append(r.request.url);r.abort()
   else:r.continue_()
  page.route('**/*',route);start=time.perf_counter();page.goto((PAGES/rel).as_uri(),wait_until='load');load_ms=(time.perf_counter()-start)*1000
  row={'input':rel,'sha256':pins[str(PAGES/rel)],'bytes':(PAGES/rel).stat().st_size,'load_ms':load_ms,'initial_dom':page.evaluate(COUNT_JS),'interactions':[]}
  page.screenshot(path=str(folder/'overview.png'))
  try:
   node=page.locator('.uf-node[data-id="'+card_id+'"]:visible').first;node.scroll_into_view_if_needed();start=time.perf_counter();node.click();card=page.locator('.uf-card-detail[data-card-id="'+card_id+'"]:visible').first;card.wait_for(state='visible');elapsed=(time.perf_counter()-start)*1000
   card.scroll_into_view_if_needed();page.evaluate('async () => {await Promise.all(document.getAnimations().map(a=>a.finished.catch(()=>{})));await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));}');data=card.evaluate("""n=>({id:n.dataset.cardId,text:n.innerText,rect:n.getBoundingClientRect().toJSON(),scrollWidth:n.scrollWidth,clientWidth:n.clientWidth,chip_nodes:[...n.querySelectorAll('[data-chip-kind]')].map(c=>({kind:c.dataset.chipKind,text:c.innerText,rect:c.getBoundingClientRect().toJSON(),style:{background:getComputedStyle(c).backgroundColor,color:getComputedStyle(c).color,border:getComputedStyle(c).border,whiteSpace:getComputedStyle(c).whiteSpace,overflowWrap:getComputedStyle(c).overflowWrap}}))})""")
   row['interactions'].append({'clicked_node':card_id,'click_to_card_ms':elapsed,'card':data,'dom':page.evaluate(COUNT_JS)});page.screenshot(path=str(folder/'card-viewport.png'))
   if data['rect']['height']<5000:card.screenshot(path=str(folder/'card.png'))
   default=[x for x in data['chip_nodes'] if x['kind']=='class_default' and x['rect']['width']>0 and x['rect']['height']>0]
   if bool(default)!=expect_default:findings.append({'case':name,'kind':'unexpected_visible_default_chip','expected':expect_default,'count':len(default)})
   if data['scrollWidth']>data['clientWidth']+1:findings.append({'case':name,'kind':'card_horizontal_overflow','scrollWidth':data['scrollWidth'],'clientWidth':data['clientWidth']})
   if name=='sdxl':
    children=card.locator('.uf-node:visible').evaluate_all('(ns)=>ns.map(n=>n.dataset.id)');row['denoiser_child_ids']=children
    selected=next((v for v in children if 'down' in v),children[0] if children else None)
    if selected:
     card.locator('.uf-node[data-id="'+selected+'"]:visible').first.click();child=page.locator('.uf-card-detail[data-card-id="'+selected+'"]:visible').first;child.wait_for(state='visible');child.scroll_into_view_if_needed();page.evaluate('async () => {await Promise.all(document.getAnimations().map(a=>a.finished.catch(()=>{})));await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));}');row['interactions'].append({'clicked_node':selected,'card_id':child.get_attribute('data-card-id'),'text':child.inner_text(),'dom':page.evaluate(COUNT_JS)});page.screenshot(path=str(folder/'nested-card.png'))
    else:findings.append({'case':name,'kind':'denoiser_has_no_visible_drill_nodes'})
  except Exception as e:findings.append({'case':name,'kind':'interaction_failure','type':type(e).__name__,'reason':str(e)});page.screenshot(path=str(folder/'failure.png'))
  row.update(page_errors=errors,console=console,blocked_requests=blocked,final_dom=page.evaluate(COUNT_JS));save(folder/'result.json',row);results[name]=row
  for error in errors:findings.append({'case':name,'kind':'javascript_error','reason':error})
  ctx.close()
 # Separate synthetic style probe: tests the four actual stylesheet classes, not four architectural claims.
 ctx=browser.new_context(viewport=versions['viewport']);page=ctx.new_page();page.route('https://**',lambda r:r.abort());page.goto((PAGES/CASES[3][1]).as_uri(),wait_until='load')
 probe=page.evaluate("""() => {const root=document.querySelector('[id^="uf-"]');const host=document.createElement('div');host.id='qa-style-probe';host.style.cssText='position:fixed;left:20px;top:20px;z-index:999999;background:white;padding:25px;width:750px';host.innerHTML='<h2>Isolated CSS probe — not model facts</h2>'+['unresolved','pending_design','class_default','symbolic'].map(k=>'<p><span class="uf-fact uf-chip-'+k+'" data-chip-kind="'+k+'">'+k+' · long_token_'+('x'.repeat(300))+'</span></p>').join('');root.append(host);return [...host.querySelectorAll('[data-chip-kind]')].map(n=>({kind:n.dataset.chipKind,background:getComputedStyle(n).backgroundColor,color:getComputedStyle(n).color,border:getComputedStyle(n).border,borderRadius:getComputedStyle(n).borderRadius,rect:n.getBoundingClientRect().toJSON()}));} """)
 page.locator('#qa-style-probe').screenshot(path=str(OUT/'four-style-probe.png'));assert len({(r['background'],r['color'],r['border']) for r in probe})==4;assert all(x['rect']['width']<=750 for x in probe);save(OUT/'four-style-probe.json',{'scope':'Synthetic4 labels appended only in an isolated browser DOM; original HTML bytes untouched. Checks actual CSS rendering, does not imply four actual fact instances.','styles':probe});ctx.close();browser.close()
assert pins=={str(PAGES/path):sha(PAGES/path) for _,path,_,_ in CASES}
save(OUT/'versions.json',versions);save(OUT/'input-pins.json',pins);save(OUT/'result.json',{'status':'PASS_BOUNDED_INTERACTIONS' if not findings else 'FINDINGS','findings':findings,'cases':{k:{'load_ms':v['load_ms'],'click_to_card_ms':v['interactions'][0]['click_to_card_ms'] if v['interactions'] else None,'page_errors':v['page_errors'],'initial_cards':v['initial_dom']['cards'],'final_cards':v['final_dom']['cards']} for k,v in results.items()},'limits':'Six saved pages in fresh headless Chrome1440x1000; external fonts blocked; browser times observed under concurrent root matrix and are not latency-baseline measurements. No proof/architecture validation, exhaustive drills, mobile/accessibility coverage or output blessing. Screenshots require independent visual inspection.'})
save(OUT/'manifest.json',{'script_sha256':sha(Path(__file__)),'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}});print((OUT/'result.json').read_text());print('manifest',sha(OUT/'manifest.json'))
