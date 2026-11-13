# SPDX-License-Identifier: Apache-2.0
# Public vLLM fused_qk_norm_rope contract, independently expressed in cuTile.
import cuda.tile as ct
@ct.kernel
def kernel(QG,K,QW,KW,Cache,Pos,QO,KO,GO):
 token=ct.bid(0);head=ct.bid(1);col=ct.arange(256,dtype=ct.int32)
 if head<16:
  x=ct.gather(QG,(token,head*512+col)).astype(ct.float32)
  w=ct.gather(QW,col).astype(ct.float32)
 else:
  x=ct.gather(K,(token,(head-16)*256+col)).astype(ct.float32)
  w=ct.gather(KW,col).astype(ct.float32)
 pair=ct.reshape(x,(128,2))
 a=ct.reshape(ct.extract(pair,(0,0),(128,1)),(128,))
 b=ct.reshape(ct.extract(pair,(0,1),(128,1)),(128,))
 sums=ct.add(ct.mul(a,a,rounding_mode=ct.RoundingMode.RN),ct.mul(b,b,rounding_mode=ct.RoundingMode.RN),rounding_mode=ct.RoundingMode.RN)
 sums=ct.reshape(sums,(4,32))
 sums=ct.add(ct.extract(sums,(0,0),(4,16)),ct.extract(sums,(0,1),(4,16)),rounding_mode=ct.RoundingMode.RN)
 sums=ct.add(ct.extract(sums,(0,0),(4,8)),ct.extract(sums,(0,1),(4,8)),rounding_mode=ct.RoundingMode.RN)
 sums=ct.add(ct.extract(sums,(0,0),(4,4)),ct.extract(sums,(0,1),(4,4)),rounding_mode=ct.RoundingMode.RN)
 sums=ct.add(ct.extract(sums,(0,0),(4,2)),ct.extract(sums,(0,1),(4,2)),rounding_mode=ct.RoundingMode.RN)
 sums=ct.add(ct.extract(sums,(0,0),(4,1)),ct.extract(sums,(0,1),(4,1)),rounding_mode=ct.RoundingMode.RN)
 sums=ct.reshape(sums,(4,))
 sums=ct.add(ct.extract(sums,(0,),(2,)),ct.extract(sums,(1,),(2,)),rounding_mode=ct.RoundingMode.RN)
 sums=ct.add(ct.extract(sums,(0,),(1,)),ct.extract(sums,(1,),(1,)),rounding_mode=ct.RoundingMode.RN)
 variance=ct.truediv(ct.reshape(sums,()),256.0,rounding_mode=ct.RoundingMode.FULL)
 inv=ct.rsqrt(variance+1e-6)
 xn=((x*inv)*(w+1.0)).astype(ct.bfloat16).astype(ct.float32)
 r=ct.arange(32,dtype=ct.int32)
 x1=ct.extract(xn,(0,),(32,));x2=ct.extract(xn,(1,),(32,))
 axis=ct.where((r%3==1)&(r<33),1,ct.where((r%3==2)&(r<30),2,0))
 pos=ct.gather(Pos,(axis,token))
 co=ct.gather(Cache,(pos,r)).astype(ct.float32);si=ct.gather(Cache,(pos,32+r)).astype(ct.float32)
 y1=x1*co-x2*si;y2=x2*co+x1*si
 if head<16:
  ct.scatter(QO,(token,head*256+col),xn.astype(ct.bfloat16),mask=col>=64)
  ct.scatter(QO,(token,head*256+r),y1.astype(ct.bfloat16))
  ct.scatter(QO,(token,head*256+32+r),y2.astype(ct.bfloat16))
  g=ct.gather(QG,(token,head*512+256+col));ct.scatter(GO,(token,head*256+col),g)
 else:
  ct.scatter(KO,(token,(head-16)*256+col),xn.astype(ct.bfloat16),mask=col>=64)
  ct.scatter(KO,(token,(head-16)*256+r),y1.astype(ct.bfloat16))
  ct.scatter(KO,(token,(head-16)*256+32+r),y2.astype(ct.bfloat16))
