# Independent translation of Apache-2.0 vLLM public causal_conv1d_update.
import cuda.tile as ct
@ct.kernel
def update_kernel(X,S,W,I):
 b=ct.bid(0);d=ct.bid(1)*256+ct.arange(256,dtype=ct.int32)
 slot=ct.gather(I,b)
 if slot!=0:
  s0=ct.gather(S,(slot,d,0));s1=ct.gather(S,(slot,d,1));s2=ct.gather(S,(slot,d,2));x=ct.gather(X,(b,d))
  w0=ct.gather(W,(d,0));w1=ct.gather(W,(d,1));w2=ct.gather(W,(d,2));w3=ct.gather(W,(d,3))
  # Explicit tap BF16 roundtrip before FP32 ordered addition.
  p0=ct.astype(ct.astype(ct.mul(ct.astype(s0,ct.float32),ct.astype(w0,ct.float32),rounding_mode=ct.RoundingMode.RN),ct.bfloat16),ct.float32)
  p1=ct.astype(ct.astype(ct.mul(ct.astype(s1,ct.float32),ct.astype(w1,ct.float32),rounding_mode=ct.RoundingMode.RN),ct.bfloat16),ct.float32)
  p2=ct.astype(ct.astype(ct.mul(ct.astype(s2,ct.float32),ct.astype(w2,ct.float32),rounding_mode=ct.RoundingMode.RN),ct.bfloat16),ct.float32)
  p3=ct.astype(ct.astype(ct.mul(ct.astype(x,ct.float32),ct.astype(w3,ct.float32),rounding_mode=ct.RoundingMode.RN),ct.bfloat16),ct.float32)
  acc=ct.add(0.0,p0,rounding_mode=ct.RoundingMode.RN)
  acc=ct.add(acc,p1,rounding_mode=ct.RoundingMode.RN);acc=ct.add(acc,p2,rounding_mode=ct.RoundingMode.RN);acc=ct.add(acc,p3,rounding_mode=ct.RoundingMode.RN)
  neg=ct.sub(0.0,acc,rounding_mode=ct.RoundingMode.RN)
  e=ct.exp2(ct.mul(neg,1.4426950408889634,rounding_mode=ct.RoundingMode.RN))
  y=ct.truediv(acc,ct.add(1.0,e,rounding_mode=ct.RoundingMode.RN),rounding_mode=ct.RoundingMode.FULL)
  ct.scatter(S,(slot,d,0),s1);ct.scatter(S,(slot,d,1),s2);ct.scatter(S,(slot,d,2),x)
  ct.scatter(X,(b,d),ct.astype(y,ct.bfloat16))
