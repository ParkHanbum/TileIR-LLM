"""Public constructor observation; retain existing owners, never allocate policy buffers.
Addresses are compared in-process only and never emitted in records.
"""
import functools,hashlib
from pathlib import Path
class Owners:
 def __init__(self):self.owners=[];self.snapshots=[];self.refs=[];self.binding=None;self.restored=False;self.failed=None
 def install(self):
  from vllm.v1.worker.gpu.buffer_utils import UvaBuffer
  source=Path(__file__).with_name('OWNERSHIP-SOURCE.json')
  import json
  pin=json.loads(source.read_text());assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
  original=UvaBuffer.__init__
  @functools.wraps(original)
  def observed(owner,*args,**kwargs):
   if self.failed is not None:raise RuntimeError('Latched public owner construction failure')
   try:
    result=original(owner,*args,**kwargs)
    assert result is None and len(self.owners)<1024
    self.capture(owner);return result
   except BaseException as e:
    if self.failed is None:self.failed=repr(e)
    raise
  self.binding=(UvaBuffer,original,observed);UvaBuffer.__init__=observed
 def identity(self,t):
  st=t.untyped_storage()
  return (id(t),id(st),st.data_ptr(),st.nbytes(),t.data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device))
 def capture(self,owner):
  cpu,view=owner.cpu,owner._uva
  assert cpu.device.type=='cpu' and cpu.is_pinned()
  assert cpu.data_ptr()==view.data_ptr() and cpu.untyped_storage().data_ptr()==view.untyped_storage().data_ptr(),'Observed UVA base mismatch'
  assert cpu.untyped_storage().nbytes()==view.untyped_storage().nbytes()
  self.owners.append(owner);self.snapshots.append((cpu,view,self.identity(cpu),self.identity(view),cpu.untyped_storage(),view.untyped_storage()))
 def retain(self,values,required=()):
  assert self.failed is None,'Owner constructor failure latched'
  rows={}
  for name,t in values.items():
   self.refs.append(t);st=t.untyped_storage();start=st.data_ptr();end=start+st.nbytes();matches=[]
   for i,owner in enumerate(self.owners):
    cpu,view,cpu_sig,view_sig,cpu_storage,view_storage=self.snapshots[i]
    assert owner.cpu is cpu and owner._uva is view and self.identity(cpu)==cpu_sig and self.identity(view)==view_sig,'Public owner backing changed'
    storage=view.untyped_storage();base=storage.data_ptr();limit=base+storage.nbytes()
    if st.nbytes() and base<=start and end<=limit:
     assert owner.cpu.device.type=='cpu' and owner.cpu.is_pinned()
     assert owner.cpu.numel()*owner.cpu.element_size()==view.numel()*view.element_size()
     matches.append(i)
   if name in required:assert matches,('Unbound required public UVA owner',name)
   rows[name]=dict(public_uva_owner_matches=len(matches),public_uva_owner_ordinals=matches,extent_within_owner=bool(matches),tensor_strong_ref=True,classification='retained_public_UvaBuffer' if matches else 'tensor_retained_backing_not_classified')
  return rows
 def close(self):
  if self.binding:
   cls,old,new=self.binding;assert cls.__init__ is new,'Foreign constructor change';cls.__init__=old;self.binding=None
  self.restored=True
 def release(self,synchronized):
  assert synchronized and self.restored
  n=len(self.owners);self.refs.clear();self.owners.clear();self.snapshots.clear();return dict(released=True,owners=n,after_synchronization=True)
