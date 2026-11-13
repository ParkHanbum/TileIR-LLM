from batch_ptr_int_cast_v1.qualification import Qualifier

def synthetic(torch):
 q=Qualifier(torch)
 try:
  values=[-2**63,2**63-1,-2**32,2**32,-1,0,1,2**32+17]
  src=torch.empty((224,),dtype=torch.int64,device='cpu',pin_memory=True)
  src.copy_(torch.tensor(values*28,dtype=torch.int64))
  backing=torch.full((2048,),-123456789,dtype=torch.int32,device='cuda')
  q.start('shadow');q.call(backing[:224],src);q.finish()
  backing.fill_(987654321)
  q.start('replace');q.call(backing[:224],src);q.finish()
  return dict(patterns=8,rows=224,phases=2,complete=True)
 finally:
  # Even a failed comparison/launch attempts completion before local owners leave.
  # A synchronization failure is fatal, never converted into a successful proof.
  q.transport.synchronize_and_release()
