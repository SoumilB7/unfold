"""Exact installed root source port review, static records only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path.cwd()))
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_primary_ports import read_primary_regions
path=Path('/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/diffusers/models/unets/unet_2d_condition.py')
fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
assert fingerprint=='052506ca0503a06657cb1816c278f7b23520ca22b3a74a6ac63250a6cc711a26'
index=build_program_index(SourceBundle(source='path',files=(str(path),),component_files={'root':(str(path),)}))
forward=next(row for row in index.callables if row.symbol.qualified_name=='UNet2DConditionModel.forward')
rows,spans=read_primary_regions(index,forward,'sample')
assert len(rows)==9
assert all(row['receives_previous_state'] for row in rows)
print(json.dumps({'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':fingerprint,'region_count':len(rows),'regions':rows,'scope':'exact root syntactic port routes only; no runtime target binding or execution closure'},indent=2,sort_keys=True))
