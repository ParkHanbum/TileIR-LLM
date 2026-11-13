# Public indexing/masked-fill semantics; no proprietary kernel source.
import cuda.tile as ct
@ct.kernel
def gather_kernel(S,I,O):
 r=ct.bid(0);x=ct.bid(1)*1024+ct.arange(1024,dtype=ct.int32)
 h=O.shape[1];k=O.shape[2];v=O.shape[3];idx=ct.gather(I,(r,));idx=ct.where(idx<0,idx+S.shape[0],idx)
 z=ct.gather(S,(idx,x//(k*v),(x//v)%k,x%v),mask=x<h*k*v,padding_value=0,check_bounds=False)
 ct.scatter(O,(r,x//(k*v),(x//v)%k,x%v),z,mask=x<h*k*v,check_bounds=False)
@ct.kernel
def mask_kernel(S,M):
 r=ct.bid(0);x=ct.bid(1)*1024+ct.arange(1024,dtype=ct.int32);h=S.shape[1];k=S.shape[2];v=S.shape[3]
 keep=ct.gather(M,(r,));zero=ct.full((1024,),0,dtype=ct.float32)
 ct.scatter(S,(r,x//(k*v),(x//v)%k,x%v),zero,mask=(x<h*k*v)&(~keep),check_bounds=False)
@ct.kernel
def scatter_kernel(S,I,V):
 r=ct.bid(0);x=ct.bid(1)*1024+ct.arange(1024,dtype=ct.int32);h=V.shape[1];k=V.shape[2];v=V.shape[3]
 idx=ct.gather(I,(r,));idx=ct.where(idx<0,idx+S.shape[0],idx)
 z=ct.gather(V,(r,x//(k*v),(x//v)%k,x%v),mask=x<h*k*v,padding_value=0,check_bounds=False)
 ct.scatter(S,(idx,x//(k*v),(x//v)%k,x%v),z,mask=x<h*k*v,check_bounds=False)
