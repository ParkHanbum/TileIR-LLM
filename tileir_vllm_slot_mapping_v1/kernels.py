# SPDX-License-Identifier: Apache-2.0
# Public algorithm: vLLM _compute_slot_mappings_kernel, Copyright contributors to vLLM.
# Independent TileIR CP_SIZE1 expression. Public license in public_sources/LICENSE.
import cuda.tile as ct
@ct.kernel
def slot_group(table,indices,starts,positions,sizes,enabled,out,group,requests,BLOCK:ct.Constant[int]):
 row=ct.bid(0)
 if row==requests:
  actual=ct.gather(starts,row)
  for base in range(actual,out.shape[1],BLOCK):
   col=base+ct.arange(BLOCK,dtype=ct.int32)
   ct.scatter(out,(group,col),ct.full((BLOCK,),-1,ct.int64),mask=col<out.shape[1])
 else:
  req=ct.gather(indices,row);begin=ct.gather(starts,row);end=ct.gather(starts,row+1)
  bs=ct.gather(sizes,group);en=ct.gather(enabled,group)
  for base in range(begin,end,BLOCK):
   col=base+ct.arange(BLOCK,dtype=ct.int32)
   pos=ct.gather(positions,col,mask=col<end,padding_value=0)
   blockindex=ct.where(en,pos//bs,0)
   block=ct.gather(table,(req,blockindex),mask=(req>=0)&(col<end),padding_value=0)
   # Match public source int32 block-number product before int64 position addition.
   slot=(block*bs).astype(ct.int64)+pos%bs
   slot=ct.where(en&(req>=0),slot,-1)
   ct.scatter(out,(group,col),slot,mask=col<end)
