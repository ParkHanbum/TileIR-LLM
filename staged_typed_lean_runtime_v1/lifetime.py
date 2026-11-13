"""Retain actual typed destination and explicit public optional UVA owner only."""
from staged_ownership_runtime_v1.ownership import ViewLease
class Destination:
 def __init__(self,owner):
  self.clear=type(owner).clear_staged_writes;self.lease=ViewLease.capture(owner);self.owner=owner;self.uva=getattr(owner,'_uva_buf',None);self.refs=[];self.signatures=[];self.destination_sig=self.sig(owner.gpu)
  if self.uva is not None:
   cpu,view=self.uva.cpu,self.uva._uva
   assert cpu.device.type=='cpu'and cpu.is_pinned()and view is owner.gpu
   assert cpu.data_ptr()==view.data_ptr()and cpu.untyped_storage().data_ptr()==view.untyped_storage().data_ptr()
   assert cpu.untyped_storage().nbytes()==view.untyped_storage().nbytes()
   self.refs=[cpu,view,cpu.untyped_storage(),view.untyped_storage()];self.signatures=[self.sig(cpu),self.sig(view)]
 def sig(self,t):
  s=t.untyped_storage();return(id(t),id(s),s.data_ptr(),s.nbytes(),t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.storage_offset(),str(t.dtype),str(t.device))
 def check(self):
  assert type(self.owner).clear_staged_writes is self.clear
  assert self.sig(self.owner.gpu)==self.destination_sig,'Destination identity/address changed'
  self.lease.check();assert getattr(self.owner,'_uva_buf',None)is self.uva
  if self.uva is not None:
   assert self.uva.cpu is self.refs[0]and self.uva._uva is self.refs[1]
   assert [self.sig(self.uva.cpu),self.sig(self.uva._uva)]==self.signatures
  return dict(typed_destination_retained=True,explicit_public_uva_owner=self.uva is not None,physical_backing_complete=False)
