# SPDX-License-Identifier: Apache-2.0
# Independent public zero-store contract draft. Not compiled or executed.
import cuda.tile as ct
@ct.kernel
def zero_words(out):
 i=ct.bid(0)*1024+ct.arange(1024,dtype=ct.int32)
 ct.scatter(out,i,ct.full((1024,),0,dtype=ct.int32),check_bounds=True)
