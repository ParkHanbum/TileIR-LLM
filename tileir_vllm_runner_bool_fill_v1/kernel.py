# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def fill_false(view):
 i=ct.bid(0)*256+ct.arange(256,dtype=ct.int32)
 ct.scatter(view,i,ct.full((256,),False,ct.bool_),check_bounds=True)
