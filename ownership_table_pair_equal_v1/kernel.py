# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def pair_equal(a,b,out,n:ct.Constant[int]):
 i=ct.arange(4,dtype=ct.int32)
 av=ct.gather(a,i,check_bounds=True);bv=ct.gather(b,i,check_bounds=True)
 bad=ct.where(i<n,ct.where(av==bv,0,1),0).astype(ct.int32)
 ct.scatter(out,0,ct.sum(bad,0),check_bounds=True)
