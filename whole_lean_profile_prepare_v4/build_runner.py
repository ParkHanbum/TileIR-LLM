"""Stdlib-only source-preserving instrumentation of the frozen v7 request.
Never imports/executes the generated runtime, frameworks, or target source.
"""
import ast,copy,hashlib,json,textwrap
from pathlib import Path
R=Path(__file__).resolve().parent
V5=R.parent/'whole_lean_extensions_runtime_v7'
PINS_SHA='86f60d89027d4649cee1e9d7d11eb77f10e3eaea4556ac463f89407db05c9789'
READY_SHA='af0ce882c13b1c66c0a98082652a0012515255d86aa4f0072a661c5ee51f02fe'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
def phase(name,body):
 ctx=ast.parse("torch.profiler.record_function('request_"+name+"::'+mode)",mode='eval').body
 return ast.With(items=[ast.withitem(context_expr=ctx)],body=body)
def transformed(source):
 tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 reqs=[n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='request'];assert len(reqs)==1
 old=reqs[0];body=old.body
 start=next(i for i,n in enumerate(body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='key')
 proof=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='tokens' for t in n.targets))
 assert isinstance(body[-1],ast.Return) and dump(body[-1].value)==dump(ast.Name(id='item',ctx=ast.Load()))
 assert start>0 and proof>start
 install=copy.deepcopy(body[:start]);execution=copy.deepcopy(body[start:proof]);checks=copy.deepcopy(body[proof:-1])
 assert isinstance(execution[0],ast.For) and any(isinstance(n,ast.While) for n in execution)
 assert any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.cuda.synchronize' for n in execution)
 context=ast.parse("torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False,with_flops=False)",mode='eval').body
 phases=[phase('install',install),phase('execution',execution),phase('proof',checks),phase('restore',[ast.Pass()])]
 wrapped=ast.With(items=[ast.withitem(context_expr=context,optional_vars=ast.Name(id='_request_prof',ctx=ast.Store()))],body=phases)
 tail=ast.parse("""
_trace_path=a.output/(mode+'.trace.json')
_request_prof.export_chrome_trace(str(_trace_path))
from whole_lean_profile_prepare_v4.runtime import summarize_trace
_profile_summary=summarize_trace(_trace_path,mode)
item['profile_partition']=_profile_summary
item['profiled']=True
item['whole_ms']=None
save()
assert _profile_summary['denominator_ready'],'Unresolved request kernel denominator'
return item
""").body
 new=copy.deepcopy(old);new.body=[wrapped]+tail;ast.fix_missing_locations(new)
 # Exact original numerical/proof AST statements, same order, recovered from scopes.
 flattened=sum([w.body for w in phases[:3]],[])+[tail[-1]]
 assert [dump(n) for n in flattened]==[dump(n) for n in old.body]
 lines=source.splitlines(keepends=True)
 generated=''.join(lines[:old.lineno-1])+textwrap.indent(ast.unparse(new),'  ')+'\n'+''.join(lines[old.end_lineno:])
 # Outside the one request function, AST must remain byte-for-byte structurally equal.
 gt=ast.parse(generated)
 def erase(t):
  t=copy.deepcopy(t)
  for n in ast.walk(t):
   if isinstance(n,ast.FunctionDef) and n.name=='request':n.body=[ast.Pass()]
  return dump(t)
 assert erase(gt)==erase(tree)
 # Admission occurs before the original root imports/preflight, not merely before Torch.
 marker="sys.path.insert(0,str(Path(__file__).resolve().parent.parent))\n"
 assert generated.count(marker)==1
 gate="if '--run-gpu' in sys.argv:\n from whole_lean_profile_prepare_v4.admission import require_ready\n require_ready()\n"
 generated=generated.replace(marker,marker+gate,1)
 return generated,dict(request_original_ast_sha256=hashlib.sha256(dump(old).encode()).hexdigest(),request_transformed_ast_sha256=hashlib.sha256(dump(new).encode()).hexdigest(),original_statements_preserved=True,outer_lifecycle_ast_unchanged=True,original_request_return_preserved=True,api_scope_wrappers_added=False)
def main():
 assert digest(V5/'PINS.json')==PINS_SHA and digest(V5/'READY.json')==READY_SHA
 pins=json.loads((V5/'PINS.json').read_text());src=V5/'run.py';assert digest(src)==pins[str(src)]
 generated,record=transformed(src.read_text())
 target=R/'run.py';assert not target.exists(),'Never overwrite a prepared runner'
 target.write_text(generated)
 record.update(original_path=str(src),original_sha256=digest(src),source_pins_sha256=PINS_SHA,source_ready_sha256=READY_SHA,generated_sha256=digest(target),gpu_ready=False,actual_endpoint_pending=True)
 (R/'RUNNER-SOURCE.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
