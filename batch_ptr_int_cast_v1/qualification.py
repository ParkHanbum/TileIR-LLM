# CPU-prepared interface. Every torch operation below executes only on explicit caller use.
import struct
from .api import Transport
from .contract import check,low32,descriptor
class Qualifier:
 def __init__(self,torch,save=lambda:None):self.torch=torch;self.save=save;self.transport=Transport();self.failed=False;self.mode=None;self.shadow_pass=False;self.rows=[];self.current=None;self.binding=None
 def raw(self,x):return x.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()
 def owner(self,x):return self.torch.empty(0,dtype=self.torch.int32,device=x.device).set_(x.untyped_storage(),0,(2048,),(1,))
 def start(self,mode):
  if self.failed or mode not in ('shadow','replace') or (mode=='replace'and not self.shadow_pass):self.failed=True;raise RuntimeError('phase/latch')
  self.mode=mode;self.rows=[];self.current=None
 def mark_failed(self):self.failed=True
 def call(self,dst,src,non_blocking=True):
  if self.failed or self.transport.failed or self.rows or self.mode is None:self.failed=True;raise RuntimeError('latch/cap')
  self.current=dict(mode=self.mode,native_attempts=0,native_returns=0,candidate_attempts=0,candidate_returns=0);self.save()
  try:
   check(dst,src,non_blocking);before_src=self.raw(src);before_dst=self.raw(self.owner(dst));meta=(descriptor(src),descriptor(dst));stream=self.torch.cuda.current_stream(dst.device).cuda_stream
   expected=bytearray(before_dst);expected[:896]=struct.pack('<224i',*[low32(v)for v in struct.unpack('<224q',before_src)])
   scratch=self.owner(dst).clone();private=scratch[:224]
   self.current['native_attempts']+=1;self.save()
   native_return=dst.copy_(src,non_blocking=True) if self.mode=='shadow' else private.copy_(src,non_blocking=True)
   self.current['native_returns']+=1
   self.current['candidate_attempts']+=1;self.save()
   candidate_return=self.transport.cast_into(private,src,non_blocking=True) if self.mode=='shadow' else self.transport.cast_into(dst,src,non_blocking=True)
   self.current['candidate_returns']+=1;self.transport.synchronize_and_release()
   if native_return is not (dst if self.mode=='shadow'else private) or candidate_return is not (private if self.mode=='shadow'else dst):raise ValueError('return alias')
   if self.raw(self.owner(dst))!=bytes(expected) or self.raw(scratch)!=bytes(expected):raise ValueError('full destination raw')
   if self.raw(src)!=before_src or (descriptor(src),descriptor(dst))!=meta:raise ValueError('readonly source/metadata')
   if self.torch.cuda.current_stream(dst.device).cuda_stream!=stream:raise ValueError('stream')
   self.rows.append(dict(mode=self.mode,raw=True,source_readonly=True,outside_preserved=True,native_original_calls=1 if self.mode=='shadow'else 0,native_private_oracles=0 if self.mode=='shadow'else 1,candidate_calls=1,transport=dict(self.transport.ledger[-1])));self.save();return dst
  except BaseException:self.failed=True;self.save();raise
 def finish(self):
  if self.failed or len(self.rows)!=1:self.failed=True;raise RuntimeError('census')
  if self.mode=='shadow':self.shadow_pass=True
  return dict(mode=self.mode,calls=1,rows=list(self.rows),scope='Native private oracle retained in replacement; transport memcpy not TileIR')
 def close(self):
  if self.binding:self.binding.close()
  self.transport.synchronize_and_release()

def synthetic(torch):
 q=Qualifier(torch);patterns=[-2**63,2**63-1,-2**32,2**32,-1,0,1,2**32+17]
 src=torch.empty((224,),dtype=torch.int64,device='cpu',pin_memory=True)
 src.copy_(torch.tensor(patterns*28,dtype=torch.int64))
 backing=torch.full((2048,),-123456789,dtype=torch.int32,device='cuda');q.start('shadow');q.call(backing[:224],src);q.finish()
 # A distinct current preimage intentionally differs from the shadow request.
 backing.fill_(987654321);q.start('replace');q.call(backing[:224],src);q.finish();q.close()
 return dict(patterns=8,rows=224,phases=2,complete=True)
