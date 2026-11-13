# SPDX-License-Identifier: Apache-2.0
import ast,__future__
from runtime_metadata_five_v1 import binding as meta
from batch_ptr_int_cast_v1.binding import OLD

def conv_tree(source):
 tree=meta.transform(source,'conv');hits=[]
 class Rewrite(ast.NodeTransformer):
  def visit_Call(self,n):
   if ast.dump(n,include_attributes=False)==OLD:
    hits.append(1);return ast.copy_location(ast.parse('_batch_ptr_cast(batch_ptr[0:mlist_len],mlist,non_blocking=True)',mode='eval').body,n)
   return self.generic_visit(n)
 tree=ast.fix_missing_locations(Rewrite().visit(tree))
 if len(hits)!=1:raise ValueError('exact batch-copy expression count')
 return tree

def conv_copy(original,full,cast):
 tree=conv_tree(meta.checked(original,'conv'));ns=dict(original.__globals__);ns.update(_meta_full=full,_batch_ptr_cast=cast)
 exec(compile(tree,'<public-metadata-batch-cast>','exec',flags=__future__.annotations.compiler_flag),ns)
 new=ns[original.__name__];new.__defaults__=original.__defaults__;new.__kwdefaults__=original.__kwdefaults__;new.__annotations__=original.__annotations__;return new

class Bridge:
 def __init__(self,metadata,cast):
  self.metadata=metadata;self.cast=cast;self.failed=False;self.installed=[];self.links=None
 def mark_failed(self):self.failed=True;self.metadata.failed=True;self.cast.failed=True
 def guard(self,fn):
  def wrapped(*a,**kw):
   if self.failed or self.metadata.failed or self.cast.failed or self.cast.transport.failed:raise RuntimeError('joint entry latch')
   try:return fn(*a,**kw)
   except BaseException:self.mark_failed();raise
  return wrapped
 def install(self,common,gdn,original_conv):
  # Each standalone proof is necessary but not a combined endpoint proof.
  from runtime_metadata_five_lean_v1.gates import qualification as meta_proof
  from batch_ptr_int_cast_lean_v1.gates import qualification as cast_proof
  meta_proof();cast_proof()
  if self.installed or self.metadata.binding is not None or self.cast.binding is not None:raise RuntimeError('already installed')
  old_common=common.__dict__['compute_num_computed_tokens'];old_gdn=gdn.__dict__['build']
  if old_gdn.__globals__.get('compute_causal_conv1d_metadata') is not original_conv:raise ValueError('actual caller globals mismatch')
  conv=conv_copy(original_conv,self.guard(self.metadata.full),self.guard(self.cast.call))
  new_common=self.guard(meta.copy(old_common,'common',{'_meta_subq':self.guard(self.metadata.subq),'_meta_subs':self.guard(self.metadata.subs)}))
  conv_entry=self.guard(conv)
  gdn_body=meta.copy(old_gdn,'gdn',{'_meta_gt':self.guard(self.metadata.gt),'compute_causal_conv1d_metadata':conv_entry})
  new_gdn=self.guard(gdn_body)
  self.links=(gdn_body,conv_entry,conv)
  try:
   for owner,name,old,new in [(common,'compute_num_computed_tokens',old_common,new_common),(gdn,'build',old_gdn,new_gdn)]:
    if owner.__dict__[name] is not old:raise RuntimeError('install collision')
    setattr(owner,name,new);self.installed.append((owner,name,old,new))
  except BaseException:
   self.mark_failed();self.restore();raise
 def active(self):
  if len(self.installed)!=2 or self.links is None:raise RuntimeError('bridge not active')
  for owner,name,old,new in self.installed:
   if owner.__dict__[name] is not new:raise RuntimeError('active wrapper identity changed')
  body,entry,conv=self.links
  if body.__globals__.get('compute_causal_conv1d_metadata') is not entry:raise RuntimeError('owned GDN conv binding changed')
  return True
 def start(self):
  if self.failed:raise RuntimeError('latched start')
  try:self.active();self.metadata.start('lean');self.cast.start('lean')
  except BaseException:self.mark_failed();raise
 def finish(self):
  if self.failed:raise RuntimeError('latched finish')
  try:
   self.active();m=self.metadata.finish();c=self.cast.finish()
   return dict(metadata=m,batch_cast=c,logical_calls=6,scope='Combined endpoint proof pending; H2D transport remains native')
  except BaseException:self.mark_failed();raise
 def restore(self):
  errors=[]
  for owner,name,old,new in reversed(self.installed):
   if owner.__dict__[name] is new:setattr(owner,name,old)
   else:errors.append(name)
  if errors:self.mark_failed();raise RuntimeError('restore collision')
  self.installed=[];self.links=None
 def close(self):
  errors=[]
  for fn in (self.restore,self.cast.close,self.metadata.close):
   try:fn()
   except BaseException as e:errors.append(e)
  if errors:self.mark_failed();raise RuntimeError('bridge cleanup failed') from errors[0]
