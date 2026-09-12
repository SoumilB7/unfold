"""Operator source operands enter one opaque boundary; no model or pytest."""
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree
sys.path.insert(0,str(Path.cwd()))
from model_unfolder.adapters.diffusor.unet_projection import _port_route_block
from model_unfolder.renderers.html.block_views.unet import build_runtime_port_route
from model_unfolder.renderers.html.views_diffusion import _stub_info
out=Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8-independent-9fe0868')
rows=[]
for kind in ('source_operation','inplace_operation'):
 route={'kind':kind,'operator':'+','operands':[{'kind':'formal','formal':'emb'},{'kind':'formal','formal':'aug'}]}
 block=_port_route_block(route,'review_operator','root.denoiser.primary_state_ports')
 svg=build_runtime_port_route({},_stub_info(),'operator_review',block)
 boundary=block['children'][-1];assert boundary['kind']=='unknown' and boundary['resolved'] is False
 assert 'Operand dispatch, internal dependence and mutation behavior remain unresolved' in boundary['description']
 assert block['detail']['argument_ids']==['review_operator__operand_0','review_operator__operand_1']
 xml=ElementTree.fromstring(svg);arrows=[node for node in xml.iter() if node.get('marker-end')]
 assert len(arrows)==3
 assert 'Operation result' in svg
 (out/(kind+'.svg')).write_text(svg)
 rows.append({'kind':kind,'block':block,'arrow_count':len(arrows)})
print(json.dumps({'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'results':rows},indent=2,sort_keys=True))
