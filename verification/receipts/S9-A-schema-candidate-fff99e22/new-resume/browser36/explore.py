from pathlib import Path
from playwright.sync_api import sync_playwright
import json
root=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/z-docs/12-design')
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
 context=browser.new_context(viewport={'width':1440,'height':1000});page=context.new_page()
 page.route('https://**',lambda r:r.abort())
 for name,path in [('bloom','S9-schema/defaults-thirtysixth/bloom/sparse.html'),('sdxl','S9-unet/schema-thirtysixth/stable-diffusion-xl-base-1-0.html')]:
  page.goto((root/path).as_uri(),wait_until='load');print(name,json.dumps(page.evaluate('''() => ({nodes:[...document.querySelectorAll('.uf-node')].filter(n=>n.getBoundingClientRect().width>0).map(n=>({id:n.dataset.id,rect:n.getBoundingClientRect().toJSON()})).slice(0,35),details:[...document.querySelectorAll('details')].map(n=>({open:n.open,text:n.querySelector('summary')?.textContent})),panels:[...document.querySelectorAll('.uf-inspect-panel')].map(n=>({depth:n.dataset.depth,rect:n.getBoundingClientRect().toJSON()})),cards:document.querySelectorAll('[data-card-id]').length})''')))
 browser.close()
