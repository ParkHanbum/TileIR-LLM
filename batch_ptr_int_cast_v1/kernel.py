# SPDX-License-Identifier: Apache-2.0
# Public signed integer truncation semantics, no vendor binary input.
import cuda.tile as ct
@ct.kernel
def low32_into(src,dst):
 i=ct.arange(256,dtype=ct.int32)
 x=ct.gather(src,i,check_bounds=True)
 # Integer narrowing retains low32 bits, interpreted as signed int32.
 ct.scatter(dst,i,x.astype(ct.int32),check_bounds=True)
