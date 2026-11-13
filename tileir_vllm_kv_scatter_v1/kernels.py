# SPDX-License-Identifier: Apache-2.0
# Public vLLM nonquantized KV scatter semantics; independently expressed in cuTile.
import cuda.tile as ct
@ct.kernel
def scatter(K,V,KC,VC,S,BS:ct.Constant[int]):
 t=ct.bid(0);h=ct.bid(1);d=ct.arange(256,dtype=ct.int32)
 slot=ct.gather(S,t,check_bounds=False);block=slot//BS;offset=slot%BS
 key=ct.gather(K,(t,h,d),check_bounds=False);value=ct.gather(V,(t,h,d),check_bounds=False)
 ct.scatter(KC,(block,offset,h,d),key,check_bounds=False)
 ct.scatter(VC,(block,offset,h,d),value,check_bounds=False)
