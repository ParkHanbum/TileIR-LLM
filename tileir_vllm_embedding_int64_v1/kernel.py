# SPDX-License-Identifier: Apache-2.0
# Independent implementation using public cuda.tile operations.
import cuda.tile as ct
@ct.kernel
def widen(x,y):
 i=ct.bid(0)*256+ct.arange(256,dtype=ct.int32)
 value=ct.gather(x,i,check_bounds=True).astype(ct.int64)
 ct.scatter(y,i,value,check_bounds=True)
