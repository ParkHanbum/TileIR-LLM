"""Called only by a coordinator-owned GPU worker; no imports at module load."""
from sampling_guard_synthetic_qualification_v1.cases import cases,expected
def run(torch,save,result):
 from public_sampling_guard_v1.api import classify,selected_columns_checked
 def raw(t):return t.detach().view(torch.uint8).cpu().numpy().tobytes()
 def signed(w,bits):return w if w<(1<<(bits-1))else w-(1<<bits)
 for number,c in enumerate(cases()):
  bf=c['dtype']=='torch.bfloat16';bits=16 if bf else 32;it=torch.int16 if bf else torch.int32;ft=torch.bfloat16 if bf else torch.float32
  n=c['rows']*c['cols'];o=c['offset']
  owner=torch.full((o+n+3,),signed(0x7fc1 if bf else 0x7fc00001,bits),dtype=it,device='cuda')
  words=owner[o:o+n];words.fill_(signed(c['fill'],bits))
  for i,w in c['changes']:words[i]=signed(w,bits)
  x=words.view(ft).reshape(c['rows'],c['cols']);before=raw(owner);stream=torch.cuda.current_stream().cuda_stream
  row=dict(kind='classification',case=number,spec=c,native=[],candidate=[],readonly_checks=0,passed=False);result['records'].append(row);save()
  native={'has_nan':lambda:bool(torch.isnan(x).any()),'has_posinf':lambda:bool(torch.isposinf(x).any()),'rows_have_finite':lambda:bool(torch.isfinite(x).any(dim=1).all())}[c['mode']]
  for label,fn in [('native',native),('candidate',lambda:classify(x,c['mode']))]:
   for _ in range(2):
    value=fn();row[label].append(value)
    assert type(value)is bool and value==expected(c),'Boolean class differs'
    assert raw(owner)==before and torch.cuda.current_stream().cuda_stream==stream
    row['readonly_checks']+=1
  row['passed']=True;save()
 # Exact gather payload test: repeated/edge IDs and quiet/signaling NaN words.
 for bf in (False,True):
  bits=16 if bf else 32;it=torch.int16 if bf else torch.int32;ft=torch.bfloat16 if bf else torch.float32
  payload=[0,1,0x8000,0x7f81,0xffc1]if bf else[0,1,0x80000000,0x7f800001,0xffc00001]
  owner=torch.tensor([signed(w,bits)for w in payload*4],dtype=it,device='cuda');x=owner.view(ft).reshape(4,5)
  ids=torch.tensor([0,4,3,3],dtype=torch.int64,device='cuda');before=raw(owner);ib=raw(ids);stream=torch.cuda.current_stream().cuda_stream
  row=dict(kind='gather',dtype=str(ft),native=[],candidate=[],readonly_checks=0,passed=False);result['records'].append(row);save()
  expected_bytes=b''.join((payload[i]).to_bytes(bits//8,'little')for i in [0,4,3,3])
  for label,fn in [('native',lambda:x.gather(1,ids[:,None])),('candidate',lambda:selected_columns_checked(x,ids))]:
   for _ in range(2):
    y=fn();assert tuple(y.shape)==(4,1)and y.dtype==ft and y.device==x.device and y.is_contiguous()
    assert y.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
    assert raw(y)==expected_bytes and raw(owner)==before and raw(ids)==ib
    assert torch.cuda.current_stream().cuda_stream==stream
    row[label].append(True);row['readonly_checks']+=1
  row['passed']=True;save()
 return result
