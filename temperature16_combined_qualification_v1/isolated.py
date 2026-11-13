"""Small boundary proof runs before engine creation; no automatic retries."""
WORDS=(0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x7f800001,0xff800001,0x3f800000)
def cases():
 out=[dict(name='word_'+hex(w),n=32,offset=0,word=w,ids=[31,0,31,1])for w in WORDS]
 out +=[dict(name='tail_'+str(n),n=n,offset=3,word=0x80000000,ids=[n-1,0,n-1])for n in(1,31,32,33,1023,1024)]
 out +=[dict(name='empty_'+str(n),n=n,offset=0,word=0,ids=[])for n in(0,32)]
 out +=[dict(name='bad_'+str(i),n=32,offset=0,word=0,ids=[i],reject=True)for i in(-1,32,-2**63,2**63-1)]
 return out

def run():
 import torch
 from public_temperature_zero_v1.api import selected_1d_checked,selected_all_zero
 from public_eq_all_guard_v2.api import eq_all
 rows=[]
 for stream_index in range(2):
  stream=torch.cuda.current_stream()if stream_index==0 else torch.cuda.Stream()
  with torch.cuda.stream(stream):
   for spec in cases():
    raw=torch.full((spec['offset']+spec['n']+3,),0x3f800000,dtype=torch.uint32,device='cuda')
    raw[spec['offset']:spec['offset']+spec['n']]=spec['word'];x=raw.view(torch.float32)[spec['offset']:spec['offset']+spec['n']]
    id_owner=torch.tensor([99]+spec['ids']+[99],dtype=torch.int64,device='cuda');ids=id_owner[1:-1]
    before=raw.cpu();ib=id_owner.cpu();pointer=torch.cuda.current_stream().cuda_stream
    if spec.get('reject'):
     try:selected_all_zero(x,ids)
     except ValueError:pass
     else:raise AssertionError('Invalid index accepted')
    else:
     expected=x.index_select(0,ids);native=bool((expected==0).all())
     for repeat in range(2):
      y=selected_1d_checked(x,ids)
      assert torch.equal(expected.view(torch.uint8),y.view(torch.uint8))
      assert y.dtype==x.dtype and y.shape==ids.shape and y.device==x.device and y.is_contiguous()
      if y.numel():assert y.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
      assert eq_all(y,0)is native and selected_all_zero(x,ids)is native
    assert torch.equal(raw.cpu(),before)and torch.equal(id_owner.cpu(),ib)
    assert torch.cuda.current_stream().cuda_stream==pointer
    rows.append(dict(name=spec['name'],stream_index=stream_index,passed=True,rejected=spec.get('reject',False),readonly=True))
   # FP32-only equality rejects BF16 before any dependent gather.
   bf=torch.zeros((32,),dtype=torch.bfloat16,device='cuda');ids=torch.tensor([0],dtype=torch.int64,device='cuda')
   try:selected_all_zero(bf,ids)
   except ValueError:pass
   else:raise AssertionError('BF16 EQ accepted')
   rows.append(dict(name='bf16_eq_reject',stream_index=stream_index,passed=True,rejected=True,readonly=True))
  stream.synchronize()
 assert len(rows)==52
 return dict(complete=True,cases=len(rows),rows=rows,streams=2,scope='FP32 raw gather/EQ0 boundaries and invalid inputs; BF16 equality unsupported')
