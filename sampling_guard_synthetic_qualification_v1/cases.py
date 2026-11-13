from public_sampling_guard_v1.contract import masks
def cases():
 out=[]
 for dtype in ('torch.float32','torch.bfloat16'):
  a,inf=masks(dtype);s=a+1
  words=[0,s,1,s|1,inf-1,s|(inf-1),inf,s|inf,inf+1,s|(inf+1),a,s|a,0x3f800000 if dtype=='torch.float32'else 0x3f80]
  for mode in ('has_nan','has_posinf','rows_have_finite'):
   for w in words:out.append(dict(dtype=dtype,rows=2,cols=1,offset=2,fill=0,changes=[[0,w]],mode=mode))
   for n in (1023,1024,1025):
    out.append(dict(dtype=dtype,rows=2,cols=n,offset=1,fill=0,changes=[[2*n-1,inf+1]],mode=mode))
   out.append(dict(dtype=dtype,rows=32,cols=248320,offset=0,fill=0,changes=[[32*248320-1,inf+1]],mode=mode))
   out.append(dict(dtype=dtype,rows=0,cols=3,offset=0,fill=0,changes=[],mode=mode))
   out.append(dict(dtype=dtype,rows=2,cols=0,offset=0,fill=0,changes=[],mode=mode))
  out.append(dict(dtype=dtype,rows=2,cols=3,offset=1,fill=s|inf,changes=[[0,0],[5,1]],mode='rows_have_finite'))
  out.append(dict(dtype=dtype,rows=2,cols=3,offset=1,fill=s|inf,changes=[[0,0]],mode='rows_have_finite'))
 return out
def expected(c):
 a,inf=masks(c['dtype']);rows,cols=c['rows'],c['cols'];n=rows*cols;changes=dict(c['changes']);mode=c['mode'];fill=c['fill']
 def yes(w):
  if mode=='has_nan':return (w&a)>inf
  if mode=='has_posinf':return w==inf
  return (w&a)<inf
 if mode!='rows_have_finite':return (n>len(changes)and yes(fill))or any(yes(w)for w in changes.values())
 counts=[cols if yes(fill)else 0 for _ in range(rows)]
 for i,w in changes.items():counts[i//cols]+=int(yes(w))-int(yes(fill))
 return all(v>0 for v in counts)
