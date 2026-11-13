# SPDX-License-Identifier: Apache-2.0
# Independent equality predicate using documented TileIR comparison/reduction.
import cuda.tile as ct
@ct.kernel
def partial_eq(x,partial,n:ct.Constant[int],value:ct.Constant[int]):
 i=ct.bid(0)*1024+ct.arange(1024,dtype=ct.int32)
 a=ct.gather(x,i,check_bounds=True)
 bits=ct.bitcast(a,ct.uint32)
 if value==0:
  equal=(bits & 0x7fffffff)==0
 else:
  equal=bits==0x3f800000
 bad=ct.where(i<n,ct.where(equal,0,1),0).astype(ct.int32)
 count=ct.sum(bad,0)
 ct.scatter(partial,ct.bid(0),count,check_bounds=True)
@ct.kernel
def finish_eq(partial,out,groups:ct.Constant[int],width:ct.Constant[int]):
 i=ct.arange(width,dtype=ct.int32)
 counts=ct.gather(partial,i,check_bounds=True)
 total=ct.sum(ct.where(i<groups,counts,0),0)
 ct.scatter(out,0,total,check_bounds=True)
