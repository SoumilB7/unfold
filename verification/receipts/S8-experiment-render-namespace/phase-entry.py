from pathlib import Path
import argparse,json,sys
p=argparse.ArgumentParser(add_help=False);p.add_argument('--checkout',type=Path,required=True);p.add_argument('--scripts',type=Path,required=True);a,rest=p.parse_known_args()
sys.path[:0]=[str(a.scripts),str(a.checkout)]
import demonstrate_s8_unet as demo
from model_unfolder.diagram import Diagram
assert Path(sys.modules[Diagram.__module__].__file__).resolve().is_relative_to(a.checkout.resolve())
demo.ROOT=a.checkout
sys.argv=[str(a.scripts/'demonstrate_s8_unet.py'),*rest]
try:demo.main()
except Exception as exc:
 q=argparse.ArgumentParser(add_help=False);q.add_argument('--output',type=Path);q.add_argument('--condition');b,_=q.parse_known_args(rest)
 if b.output is not None and b.condition and (b.output/b.condition).is_dir():demo._write(b.output/b.condition/'failure.json',{'status':'failed','blessed':False,'exception_type':type(exc).__name__,'detail':str(exc)})
 raise
