from pathlib import Path
import sys,json,hashlib,collections
R=Path('/private/tmp/unfold-s8-compiler-final');P=Path('/private/tmp/unfold-s8-render-phase-c38-v2/sdxl');O=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from model_unfolder.adapters.diffusor.blocks import diffusion_loop_edges
# Pure display-data construction: no model, source reader, config resolver, or inventory.
assert {e['from']for e in diffusion_loop_edges({'text_encoders':['arbitrary A','arbitrary B']})if e['to']=='denoiser'}=={'latent','timestep','encoder_0','encoder_1'}
assert not any(e['from'].startswith('encoder_')and e['to']=='denoiser'for e in diffusion_loop_edges({'text_encoders':['arbitrary A','arbitrary B'],'suppress_conditioning_source':True}))
a=json.loads((P/'legacy/ir.json').read_text());b=json.loads((P/'ordinary/ir.json').read_text());key=lambda e:json.dumps(e,sort_keys=True)
old=collections.Counter(map(key,a['extras']['render']['loop_edges']));new=collections.Counter(map(key,b['extras']['render']['loop_edges']));removed=[json.loads(e)for e in (old-new).elements()];assert len(removed)==2 and {e['from']for e in removed}=={'encoder_0','encoder_1'};assert not(new-old)
assert not json.loads((P/'legacy/facts.json').read_text()) and not json.loads((P/'legacy/qualified-facts.json').read_text())
specs={'model_unfolder/adapters/diffusor/parser.py':[(151,169),(904,936)],'model_unfolder/adapters/diffusor/unet.py':[(186,226)],'model_unfolder/adapters/diffusor/blocks.py':[(322,328),(346,356),(459,462),(672,690)],'model_unfolder/adapters/diffusor/unet_projection.py':[(429,438)]}
source={}
for file,ranges in specs.items():
 raw=(R/file).read_bytes();lines=raw.decode().splitlines();source[file]={'sha256':hashlib.sha256(raw).hexdigest(),'snippets':['\n'.join(f'{i}: {lines[i-1]}'for i in range(lo,hi+1))for lo,hi in ranges]}
svg=json.loads((O/'svg-inventory.json').read_text());maps={n:collections.defaultdict(list)for n in svg}
for n,rows in svg.items():
 for r in rows:maps[n][r['card_id']].append(r['visual_hash'])
shared=set(maps['legacy'])&set(maps['ordinary']);changed=sorted(k for k in shared if maps['legacy'][k]!=maps['ordinary'][k]);assert changed==['<architecture>','denoiser','encoder_0','encoder_1']
result={'removed_outer_edges':removed,'edge_author':'blocks.py:322–328/346–351 derives encoder targets solely from nonempty text_encoders display list; pure presence-only reproduction passed. Parser.py:151–169 supplies independently parsed component names, not an output-to-denoiser connection proof.','legacy_modality_author':'Actual input encoder_hid_dim_type=None; parser._u11_unet_conditioning uses conventional elif has_text modality fallback. Encoder internals do not prove external edges.','image_author':'unet_geom sets video from temporal and no output_domain. blocks.py:461/672–690 defaults nonaudio/nonvideo output to Image/generated image pixels; no codec output-domain reader premise.','named_removal_verdict':'ACCEPT conservative removal of these two unproved outer connections and default output-domain assertion; owner separately accepted after source-path review. No source-proven drawing removed by these specific changes.','honesty_cost':'Supplied encoders remain visible but no longer have arrows into the denoiser. Their role/text dimension chips are limited; Image label becomes Output domain unresolved. This intentionally communicates less than the old conventional pipeline drawing.','shared_svg_cards_changed':changed,'shared_other_drill_svg_lists_identical':True,'source':source}
(O/'named-removals.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'presence_only_edge_probe':True,'actual_removed_edges':len(removed),'shared_svg_cards_changed':changed}))
