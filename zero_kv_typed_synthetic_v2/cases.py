"""Private synthetic word geometry. No addresses or actual KV payloads."""
CASES=[
 dict(name='one_word',words=9,segments=[(3,4,1,1)],ids=[0]),
 dict(name='tail_overlap_duplicate',words=2200,segments=[(2,1100,1025,2),(10,1100,1020,2)],ids=[1,0,1]),
 dict(name='empty_noop',words=37,segments=[(4,16,9,2)],ids=[]),
 dict(name='max64MiB_plus_tail',words=16777219,segments=[(1,16777217,16777217,1)],ids=[0]),
]
def intervals(case):
 out=[]
 for offset,stride,page,blocks in case['segments']:
  for i in case['ids']:
   assert type(i)is int and 0<=i<blocks
   out.append(((offset+i*stride)*4,(offset+i*stride+page)*4))
 return out
