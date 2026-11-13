# SPDX-License-Identifier: Apache-2.0
# Public1024-chunk log-softmax order; reduction/exp/log raw equality unverified.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def kernel(X,IDs,Y):
 row=ct.bid(0);col=ct.arange(1024,dtype=ct.int32)
 maximum=ct.full((),float('-inf'),ct.float32)
 for base in range(0,248320,1024):
  x=ct.gather(X,(row,base+col),check_bounds=True,padding_value=float('-inf')).astype(ct.float32)
  maximum=ct.max(ct.maximum(x,maximum),axis=0)
 total=ct.full((),0,ct.float32)
 for base in range(0,248320,1024):
  x=ct.gather(X,(row,base+col),check_bounds=True,padding_value=0).astype(ct.float32)
  e=ct.exp(ct.sub(x,maximum,rounding_mode=RM.RN),rounding_mode=RM.APPROX)
  e=ct.where(base+col<248320,e,0)
  lanes=ct.reshape(e,(128,8))
  acc=ct.extract(lanes,(0,0),(128,1))
  acc=ct.add(acc,ct.extract(lanes,(0,1),(128,1)),rounding_mode=RM.RN)
  acc=ct.add(acc,ct.extract(lanes,(0,2),(128,1)),rounding_mode=RM.RN)
  acc=ct.add(acc,ct.extract(lanes,(0,3),(128,1)),rounding_mode=RM.RN)
  acc=ct.add(acc,ct.extract(lanes,(0,4),(128,1)),rounding_mode=RM.RN)
  acc=ct.add(acc,ct.extract(lanes,(0,5),(128,1)),rounding_mode=RM.RN)
  acc=ct.add(acc,ct.extract(lanes,(0,6),(128,1)),rounding_mode=RM.RN)
  acc=ct.add(acc,ct.extract(lanes,(0,7),(128,1)),rounding_mode=RM.RN)
  acc=ct.reshape(acc,(4,32))
  acc=ct.add(ct.extract(acc,(0,0),(4,16)),ct.extract(acc,(0,1),(4,16)),rounding_mode=RM.RN)
  acc=ct.add(ct.extract(acc,(0,0),(4,8)),ct.extract(acc,(0,1),(4,8)),rounding_mode=RM.RN)
  acc=ct.add(ct.extract(acc,(0,0),(4,4)),ct.extract(acc,(0,1),(4,4)),rounding_mode=RM.RN)
  acc=ct.add(ct.extract(acc,(0,0),(4,2)),ct.extract(acc,(0,1),(4,2)),rounding_mode=RM.RN)
  acc=ct.add(ct.extract(acc,(0,0),(4,1)),ct.extract(acc,(0,1),(4,1)),rounding_mode=RM.RN)
  acc=ct.reshape(acc,(4,))
  acc=ct.add(ct.extract(acc,(0,),(2,)),ct.extract(acc,(1,),(2,)),rounding_mode=RM.RN)
  acc=ct.add(ct.extract(acc,(0,),(1,)),ct.extract(acc,(1,),(1,)),rounding_mode=RM.RN)
  total=ct.add(total,ct.reshape(acc,()),rounding_mode=RM.RN)
 logsum=ct.log(total);j=ct.arange(8,dtype=ct.int32)
 ids=ct.gather(IDs,(row,j),check_bounds=True,padding_value=0)
 x=ct.gather(X,(row,ids),check_bounds=True).astype(ct.float32)
 value=ct.sub(ct.sub(x,maximum,rounding_mode=RM.RN),logsum,rounding_mode=RM.RN)
 ct.scatter(Y,(row,j),value,check_bounds=True)
