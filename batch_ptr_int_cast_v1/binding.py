# SPDX-License-Identifier: Apache-2.0
import ast,inspect,textwrap,hashlib,json,__future__
from pathlib import Path
EXPECTED=json.loads(Path(__file__).with_name('SOURCE.json').read_text())
OLD=ast.dump(ast.parse('batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)',mode='eval').body,include_attributes=False)
def build(fn,helper):
 e=EXPECTED
 if fn.__code__.co_freevars or inspect.getsourcefile(fn)!=e['path'] or fn.__code__.co_firstlineno!=e['firstlineno']:raise ValueError('source identity')
 if hashlib.sha256(Path(e['path']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('source hash')
 s=textwrap.dedent(inspect.getsource(fn));tree=ast.parse(s)
 if ast.dump(tree,include_attributes=False)!=e['ast']:raise ValueError('source AST')
 hits=[]
 class Change(ast.NodeTransformer):
  def visit_Call(self,n):
   if ast.dump(n,include_attributes=False)==OLD:
    hits.append(1);return ast.copy_location(ast.parse('_batch_ptr_cast(batch_ptr[0:mlist_len], mlist, non_blocking=True)',mode='eval').body,n)
   return self.generic_visit(n)
 tree=ast.fix_missing_locations(Change().visit(tree))
 if len(hits)!=1:raise ValueError('target count')
 ns=dict(fn.__globals__);ns['_batch_ptr_cast']=helper;exec(compile(tree,'<public-batch-cast>','exec',flags=__future__.annotations.compiler_flag),ns)
 out=ns[fn.__name__];out.__defaults__=fn.__defaults__;out.__kwdefaults__=fn.__kwdefaults__;out.__annotations__=fn.__annotations__;return out
class Binding:
 def __init__(self,module,helper,failed,mark_failed):
  from runtime_metadata_five_v1.binding import checked
  if module.__name__!='vllm.v1.attention.backends.gdn_attn':raise ValueError('module identity')
  checked(module.GDNAttentionMetadataBuilder.build,'gdn')
  self.module=module;self.original=module.compute_causal_conv1d_metadata;self.installed=False
  body=build(self.original,helper)
  def guarded(*args,**kwargs):
   if failed():raise RuntimeError('latched parent entry')
   try:return body(*args,**kwargs)
   except BaseException:mark_failed();raise
  self.replacement=guarded
  if module.GDNAttentionMetadataBuilder.build.__globals__.get('compute_causal_conv1d_metadata') is not self.original:raise ValueError('cached/copied caller unsupported')
 def install(self):
  if self.module.compute_causal_conv1d_metadata is not self.original:raise RuntimeError('binding collision')
  self.module.compute_causal_conv1d_metadata=self.replacement;self.installed=True
 def close(self):
  if self.installed:
   if self.module.compute_causal_conv1d_metadata is not self.replacement:raise RuntimeError('foreign restore collision')
   self.module.compute_causal_conv1d_metadata=self.original;self.installed=False
