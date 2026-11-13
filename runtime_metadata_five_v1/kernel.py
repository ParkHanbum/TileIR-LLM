# SPDX-License-Identifier: Apache-2.0
# Independent public integer elementwise operations, not vendor binary analysis.
import cuda.tile as ct
@ct.kernel
def sub32(a,b,out):
 i=ct.arange(32,dtype=ct.int32)
 x=ct.gather(a,i,check_bounds=True).astype(ct.uint32)
 y=ct.gather(b,i,check_bounds=True).astype(ct.uint32)
 ct.scatter(out,i,(x-y).astype(ct.int32),check_bounds=True)
@ct.kernel
def gt32(a,out):
 i=ct.arange(32,dtype=ct.int32)
 ct.scatter(out,i,ct.gather(a,i,check_bounds=True)>0,check_bounds=True)
@ct.kernel
def full2048(out):
 i=ct.bid(0)*256+ct.arange(256,dtype=ct.int32)
 ct.scatter(out,i,ct.full((256,),-1,dtype=ct.int32),check_bounds=True)
