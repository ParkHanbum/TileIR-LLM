# SPDX-License-Identifier: Apache-2.0
from .contract import check,descriptor,N
class Transport:
 def __init__(self,torch_module=None,tile_module=None,kernel=None):
  self.torch=torch_module;self.tile=tile_module;self.kernel=kernel;self.pending=[];self.failed=False;self.ledger=[]
 def cast_into(self,dst,src,non_blocking=True):
  if self.failed:raise RuntimeError('latched transport')
  try:
   if self.pending:raise RuntimeError('one pending transport scope')
   check(dst,src,non_blocking)
   if self.torch is None:
    import torch,cuda.tile as ct
    from .kernel import low32_into
    self.torch,self.tile,self.kernel=torch,ct,low32_into
   torch,ct=self.torch,self.tile;stream=torch.cuda.current_stream(dst.device)
   before=(descriptor(src),descriptor(dst));tmp=torch.empty((N,),dtype=torch.int64,device=dst.device)
   if str(tmp.dtype)!='torch.int64' or tmp.device!=dst.device or tuple(tmp.shape)!=(N,) or tmp.untyped_storage().nbytes()!=N*8 or tmp.untyped_storage().data_ptr()==dst.untyped_storage().data_ptr():raise ValueError('transport temporary')
   row=dict(transfer_attempts=0,transfer_returns=0,cast_launch_attempts=0,cast_launch_returns=0,temporary_bytes=N*8,source_pinned=True,same_dtype_transfer=True,scope='source call ledger, not physical GPU kernel count')
   self.ledger.append(row);self.pending.append((src,tmp,dst,stream))
   row['transfer_attempts']+=1
   returned=tmp.copy_(src,non_blocking=True)
   row['transfer_returns']+=1
   if returned is not tmp or torch.cuda.current_stream(dst.device).cuda_stream!=stream.cuda_stream:raise ValueError('transfer stream/return')
   row['cast_launch_attempts']+=1;ct.launch(stream,(1,),self.kernel,(tmp,dst));row['cast_launch_returns']+=1
   if torch.cuda.current_stream(dst.device).cuda_stream!=stream.cuda_stream or (descriptor(src),descriptor(dst))!=before:raise ValueError('stream or metadata changed')
   return dst
  except BaseException:self.failed=True;raise
 def synchronize_and_release(self):
  # Keep every pending reference if even one stream synchronization fails.
  try:
   seen=set()
   for src,tmp,dst,stream in self.pending:
    key=(str(dst.device),stream.cuda_stream)
    if key not in seen:stream.synchronize();seen.add(key)
   self.pending=[]
  except BaseException:self.failed=True;raise
