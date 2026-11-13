# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,inspect,json,textwrap,__future__
from pathlib import Path
EXPECTED=json.loads(Path(__file__).with_name("SOURCE.json").read_text())
def checked(fn,key):
 e=EXPECTED[key]
 if inspect.getsourcefile(fn)!=e["path"] or fn.__code__.co_firstlineno!=e["firstlineno"] or fn.__code__.co_freevars:raise ValueError("source identity/closure")
 if hashlib.sha256(Path(e["path"]).read_bytes()).hexdigest()!=e["sha256"]:raise ValueError("source hash")
 s=textwrap.dedent(inspect.getsource(fn))
 if ast.dump(ast.parse(s),include_attributes=False)!=e["ast"]:raise ValueError("source AST")
 return s
TARGETS={"common":[("self.query_start_loc[1:] - self.query_start_loc[:-1]","_meta_subq(self.query_start_loc[1:],self.query_start_loc[:-1])"),("self.seq_lens - query_lens","_meta_subs(self.seq_lens,query_lens)")],"gdn":[("context_lens_tensor > 0","_meta_gt(context_lens_tensor)")],"conv":[("torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device)","_meta_full((MAX_NUM_PROGRAMS,),PAD_SLOT_ID,dtype=torch.int32,device=device)")]}
def transform(source,key):
 tree=ast.parse(source);hits=[0]*len(TARGETS[key]);maps=[(ast.dump(ast.parse(a,mode="eval").body,include_attributes=False),b) for a,b in TARGETS[key]]
 class Rewrite(ast.NodeTransformer):
  def visit(self,node):
   for i,(old,new) in enumerate(maps):
    if ast.dump(node,include_attributes=False)==old:hits[i]+=1;return ast.copy_location(ast.parse(new,mode="eval").body,node)
   return super().visit(node)
 tree=ast.fix_missing_locations(Rewrite().visit(tree))
 if hits!=({"common":[1,1],"gdn":[1],"conv":[2]}[key]):raise ValueError("expression count")
 return tree

def copy(fn,key,helpers):
 tree=transform(checked(fn,key),key);ns=dict(fn.__globals__);ns.update(helpers)
 exec(compile(tree,"<tileir-public-metadata>","exec",flags=__future__.annotations.compiler_flag),ns)
 new=ns[fn.__name__];new.__defaults__=fn.__defaults__;new.__kwdefaults__=fn.__kwdefaults__;new.__annotations__=fn.__annotations__;return new
class Bindings:
 def __init__(self,common,gdn,conv,q):
  self.common,self.gdn=common,gdn;self.old_common=common.__dict__["compute_num_computed_tokens"];self.old_gdn=gdn.__dict__["build"];self.installed=[]
  conv_copy=copy(conv,"conv",{"_meta_full":q.full})
  self.new_common=q.guard(copy(self.old_common,"common",{"_meta_subq":q.subq,"_meta_subs":q.subs}))
  self.new_gdn=q.guard(copy(self.old_gdn,"gdn",{"_meta_gt":q.gt,"compute_causal_conv1d_metadata":conv_copy}))
 def install(self):
  for owner,name,old,new in [(self.common,"compute_num_computed_tokens",self.old_common,self.new_common),(self.gdn,"build",self.old_gdn,self.new_gdn)]:
   if owner.__dict__[name] is not old:raise RuntimeError("binding collision")
   setattr(owner,name,new);self.installed.append((owner,name,old,new))
 def close(self):
  errors=[]
  for owner,name,old,new in reversed(self.installed):
   if owner.__dict__[name] is new:setattr(owner,name,old)
   else:errors.append(name)
  if errors:raise RuntimeError("restore collision")
  self.installed=[]
