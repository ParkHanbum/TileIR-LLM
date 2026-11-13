"""Explicit single-body combination interface, no module/Tensor patch."""
import inspect,textwrap
from pathlib import Path
P=Path(__file__).resolve().parent.parent

def attention_body(original,finite,equal_one):
 source=textwrap.dedent(inspect.getsource(original))
 baseline=(P/'floating_guard_504_design_v1/attention.original.py.txt').read_text()
 assert source.strip()==baseline.strip()
 f='bool(torch.isfinite(out).all())';e='bool((t==1).all())'
 assert source.count(f)==source.count(e)==1
 candidate=source.replace(f,"_finite('attention.out',out)").replace(e,'_equal_one(n,t)')
 ns=dict(original.__globals__);ns.update(_finite=finite,_equal_one=equal_one)
 exec(compile(candidate,'<owned-attention-finite-eq>','exec'),ns)
 return ns['apply']
