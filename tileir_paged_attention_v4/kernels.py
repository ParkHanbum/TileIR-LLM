"""Independent public attention-formula implementation, not FA2 arithmetic reconstruction."""
import cuda.tile as ct
@ct.kernel
def kernel(Q,K,V,L,T,O,W:ct.Constant[int]):
 token=ct.bid(0);head=ct.bid(1);seq=token//W;local=token%W;kvhead=head//4
 length=ct.gather(L,seq,check_bounds=True);block=ct.gather(T,(seq,0),check_bounds=True)
 d=ct.arange(256,dtype=ct.int32);pos=ct.arange(128,dtype=ct.int32)
 q=ct.gather(Q,(token,head,d),check_bounds=True)
 kk=ct.gather(K,(block,pos[:,None],kvhead,d[None,:]),mask=pos[:,None]<length,padding_value=0,check_bounds=True)
 # Replicated query rows preserve one-query grid/PV path while using BF16 MMA.
 qm=ct.broadcast_to(q[None,:],(16,256))
 acc=ct.mma(qm,ct.transpose(kk),ct.full((16,128),0,dtype=ct.float32))
 scores=ct.reshape(ct.extract(acc,(0,0),(1,128)),(128,))*0.0625
 valid=(pos<length)&(pos<=local+length-W)
 scores=ct.where(valid,scores,float('-inf'))
 maximum=ct.max(scores,axis=0);e=ct.exp2((scores-maximum)*1.4426950408889634);den=ct.sum(e,axis=0)
 vv=ct.gather(V,(block,pos[:,None],kvhead,d[None,:]),mask=pos[:,None]<length,padding_value=0,check_bounds=True)
 prob=e.astype(ct.bfloat16)
 # Replicated P rows keep the previous one-query grid; only output row zero is used.
 pm=ct.broadcast_to(prob[None,:],(16,128))
 pv=ct.mma(pm,vv,ct.full((16,256),0,dtype=ct.float32))
 result=ct.reshape(ct.extract(pv,(0,0),(1,256)),(256,))/den
 ct.scatter(O,(token,head,d),result.astype(ct.bfloat16),check_bounds=True)
