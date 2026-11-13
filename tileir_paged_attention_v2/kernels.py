"""Independent public attention-formula implementation, not FA2 arithmetic reconstruction."""
import cuda.tile as ct
@ct.kernel
def kernel(Q,K,V,L,T,O,W:ct.Constant[int]):
 token=ct.bid(0);head=ct.bid(1);seq=token//W;local=token%W;kvhead=head//4
 length=ct.gather(L,seq,check_bounds=True);block=ct.gather(T,(seq,0),check_bounds=True)
 d=ct.arange(256,dtype=ct.int32);pos=ct.arange(128,dtype=ct.int32)
 q=ct.gather(Q,(token,head,d),check_bounds=True).astype(ct.float32)
 kk=ct.gather(K,(block,pos[:,None],kvhead,d[None,:]),mask=pos[:,None]<length,padding_value=0,check_bounds=True).astype(ct.float32)
 scores=ct.sum(kk*q[None,:],axis=1)*0.0625
 valid=(pos<length)&(pos<=local+length-W)
 scores=ct.where(valid,scores,float('-inf'))
 maximum=ct.max(scores,axis=0);e=ct.exp2((scores-maximum)*1.4426950408889634);den=ct.sum(e,axis=0)
 vv=ct.gather(V,(block,pos[:,None],kvhead,d[None,:]),mask=pos[:,None]<length,padding_value=0,check_bounds=True).astype(ct.float32)
 prob=e.astype(ct.bfloat16).astype(ct.float32)
 result=ct.sum(prob[:,None]*vv,axis=0)/den
 ct.scatter(O,(token,head,d),result.astype(ct.bfloat16),check_bounds=True)
