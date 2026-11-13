# SPDX-License-Identifier: Apache-2.0
# Public algorithm: vLLM _gather_block_tables_kernel, Copyright contributors to vLLM.
# Independent TileIR expression; retains valid-prefix/tail/padded-row semantics.
import cuda.tile as ct

@ct.kernel
def gather_group(src,counts,indices,out,group,actual,BLOCK:ct.Constant[int]):
    row=ct.bid(0)
    col=ct.bid(1)*BLOCK+ct.arange(BLOCK,dtype=ct.int32)
    width=src.shape[1]
    if row>=actual:
        ct.scatter(out,(row,col),ct.full((BLOCK,),0,ct.int32),mask=col<width)
    else:
        req=ct.gather(indices,row)
        n=ct.gather(counts,(group,req))
        values=ct.gather(src,(req,col),mask=(col<n)&(col<width),padding_value=0)
        ct.scatter(out,(row,col),values,mask=(col<n)&(col<width))
