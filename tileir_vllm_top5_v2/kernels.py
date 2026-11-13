# Public PyTorch SortUtils.cuh bitonic network adaptation.
# Copyright holders and BSD license: LICENSE-PYTORCH. Other code Apache-2.0.
import cuda.tile as ct
@ct.kernel
def blocks(X,V,I):
 row=ct.bid(0);b=ct.bid(1);col=b*1024+ct.arange(1024,dtype=ct.int32)
 x=ct.gather(X,(row,col),check_bounds=True,padding_value=0)
 bits=ct.bitcast(x,ct.uint16).astype(ct.int32)
 key=bits^ct.where((bits&32768)!=0,65535,32768)
 active=col<248320
 for j in range(5):
  mx=ct.max(ct.where(active,key,-1),axis=0)
  idx=ct.min(ct.where(active&(key==mx),col,2147483647),axis=0)
  ct.scatter(V,(row,b*5+j),mx,check_bounds=True);ct.scatter(I,(row,b*5+j),idx,check_bounds=True)
  active=active&(col!=idx)
@ct.kernel
def finish(V,I,Y):
 row=ct.bid(0);p=ct.arange(2048,dtype=ct.int32);lane=ct.arange(32,dtype=ct.int32)
 key=ct.gather(V,(row,p),check_bounds=True,padding_value=-1)
 idx=ct.gather(I,(row,p),check_bounds=True,padding_value=2147483647)
 active=p<1215
 chosen_key=ct.full((32,),-1,ct.int32);chosen_idx=ct.full((32,),2147483647,ct.int32)
 kth=ct.full((),-1,ct.int32)
 for j in range(5):
  mx=ct.max(ct.where(active,key,-1),axis=0)
  ii=ct.min(ct.where(active&(key==mx),idx,2147483647),axis=0)
  chosen_key=ct.where(lane==j,mx,chosen_key);chosen_idx=ct.where(lane==j,ii,chosen_idx)
  kth=mx;active=active&(idx!=ii)
 # Reconstruct public mbtopk gather: strict-above ascending index, then cutoff equals.
 remaining=lane<5;keys=ct.full((32,),0.,ct.float32);ids=ct.full((32,),0,ct.int32)
 for j in range(5):
  above=remaining&(chosen_key>kth)
  first=ct.min(ct.where(above,chosen_idx,2147483647),axis=0)
  other=ct.min(ct.where(remaining,chosen_idx,2147483647),axis=0)
  ii=ct.where(first==2147483647,other,first)
  kk=ct.max(ct.where(remaining&(chosen_idx==ii),chosen_key,-1),axis=0)
  bits=(kk^ct.where((kk&32768)!=0,32768,65535)).astype(ct.uint16)
  value=ct.bitcast(bits,ct.bfloat16).astype(ct.float32)
  keys=ct.where(lane==j,value,keys);ids=ct.where(lane==j,ii,ids);remaining=remaining&(chosen_idx!=ii)
 valid=lane<5
 # Network size=2, stride=1
 keys3=ct.reshape(keys,(16,2,1))
 keysa=ct.extract(keys3,(0,0,0),(16,1,1));keysb=ct.extract(keys3,(0,1,0),(16,1,1))
 ids3=ct.reshape(ids,(16,2,1))
 idsa=ct.extract(ids3,(0,0,0),(16,1,1));idsb=ct.extract(ids3,(0,1,0),(16,1,1))
 valid3=ct.reshape(valid,(16,2,1))
 valida=ct.extract(valid3,(0,0,0),(16,1,1));validb=ct.extract(valid3,(0,1,0),(16,1,1))
 direction=((ct.arange(16,dtype=ct.int32)[:,None,None]*1+ct.arange(1,dtype=ct.int32)[None,None,:])&1)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=4, stride=2
 keys3=ct.reshape(keys,(8,2,2))
 keysa=ct.extract(keys3,(0,0,0),(8,1,2));keysb=ct.extract(keys3,(0,1,0),(8,1,2))
 ids3=ct.reshape(ids,(8,2,2))
 idsa=ct.extract(ids3,(0,0,0),(8,1,2));idsb=ct.extract(ids3,(0,1,0),(8,1,2))
 valid3=ct.reshape(valid,(8,2,2))
 valida=ct.extract(valid3,(0,0,0),(8,1,2));validb=ct.extract(valid3,(0,1,0),(8,1,2))
 direction=((ct.arange(8,dtype=ct.int32)[:,None,None]*2+ct.arange(2,dtype=ct.int32)[None,None,:])&2)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=4, stride=1
 keys3=ct.reshape(keys,(16,2,1))
 keysa=ct.extract(keys3,(0,0,0),(16,1,1));keysb=ct.extract(keys3,(0,1,0),(16,1,1))
 ids3=ct.reshape(ids,(16,2,1))
 idsa=ct.extract(ids3,(0,0,0),(16,1,1));idsb=ct.extract(ids3,(0,1,0),(16,1,1))
 valid3=ct.reshape(valid,(16,2,1))
 valida=ct.extract(valid3,(0,0,0),(16,1,1));validb=ct.extract(valid3,(0,1,0),(16,1,1))
 direction=((ct.arange(16,dtype=ct.int32)[:,None,None]*1+ct.arange(1,dtype=ct.int32)[None,None,:])&2)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=8, stride=4
 keys3=ct.reshape(keys,(4,2,4))
 keysa=ct.extract(keys3,(0,0,0),(4,1,4));keysb=ct.extract(keys3,(0,1,0),(4,1,4))
 ids3=ct.reshape(ids,(4,2,4))
 idsa=ct.extract(ids3,(0,0,0),(4,1,4));idsb=ct.extract(ids3,(0,1,0),(4,1,4))
 valid3=ct.reshape(valid,(4,2,4))
 valida=ct.extract(valid3,(0,0,0),(4,1,4));validb=ct.extract(valid3,(0,1,0),(4,1,4))
 direction=((ct.arange(4,dtype=ct.int32)[:,None,None]*4+ct.arange(4,dtype=ct.int32)[None,None,:])&4)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=8, stride=2
 keys3=ct.reshape(keys,(8,2,2))
 keysa=ct.extract(keys3,(0,0,0),(8,1,2));keysb=ct.extract(keys3,(0,1,0),(8,1,2))
 ids3=ct.reshape(ids,(8,2,2))
 idsa=ct.extract(ids3,(0,0,0),(8,1,2));idsb=ct.extract(ids3,(0,1,0),(8,1,2))
 valid3=ct.reshape(valid,(8,2,2))
 valida=ct.extract(valid3,(0,0,0),(8,1,2));validb=ct.extract(valid3,(0,1,0),(8,1,2))
 direction=((ct.arange(8,dtype=ct.int32)[:,None,None]*2+ct.arange(2,dtype=ct.int32)[None,None,:])&4)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=8, stride=1
 keys3=ct.reshape(keys,(16,2,1))
 keysa=ct.extract(keys3,(0,0,0),(16,1,1));keysb=ct.extract(keys3,(0,1,0),(16,1,1))
 ids3=ct.reshape(ids,(16,2,1))
 idsa=ct.extract(ids3,(0,0,0),(16,1,1));idsb=ct.extract(ids3,(0,1,0),(16,1,1))
 valid3=ct.reshape(valid,(16,2,1))
 valida=ct.extract(valid3,(0,0,0),(16,1,1));validb=ct.extract(valid3,(0,1,0),(16,1,1))
 direction=((ct.arange(16,dtype=ct.int32)[:,None,None]*1+ct.arange(1,dtype=ct.int32)[None,None,:])&4)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=16, stride=8
 keys3=ct.reshape(keys,(2,2,8))
 keysa=ct.extract(keys3,(0,0,0),(2,1,8));keysb=ct.extract(keys3,(0,1,0),(2,1,8))
 ids3=ct.reshape(ids,(2,2,8))
 idsa=ct.extract(ids3,(0,0,0),(2,1,8));idsb=ct.extract(ids3,(0,1,0),(2,1,8))
 valid3=ct.reshape(valid,(2,2,8))
 valida=ct.extract(valid3,(0,0,0),(2,1,8));validb=ct.extract(valid3,(0,1,0),(2,1,8))
 direction=((ct.arange(2,dtype=ct.int32)[:,None,None]*8+ct.arange(8,dtype=ct.int32)[None,None,:])&8)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=16, stride=4
 keys3=ct.reshape(keys,(4,2,4))
 keysa=ct.extract(keys3,(0,0,0),(4,1,4));keysb=ct.extract(keys3,(0,1,0),(4,1,4))
 ids3=ct.reshape(ids,(4,2,4))
 idsa=ct.extract(ids3,(0,0,0),(4,1,4));idsb=ct.extract(ids3,(0,1,0),(4,1,4))
 valid3=ct.reshape(valid,(4,2,4))
 valida=ct.extract(valid3,(0,0,0),(4,1,4));validb=ct.extract(valid3,(0,1,0),(4,1,4))
 direction=((ct.arange(4,dtype=ct.int32)[:,None,None]*4+ct.arange(4,dtype=ct.int32)[None,None,:])&8)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=16, stride=2
 keys3=ct.reshape(keys,(8,2,2))
 keysa=ct.extract(keys3,(0,0,0),(8,1,2));keysb=ct.extract(keys3,(0,1,0),(8,1,2))
 ids3=ct.reshape(ids,(8,2,2))
 idsa=ct.extract(ids3,(0,0,0),(8,1,2));idsb=ct.extract(ids3,(0,1,0),(8,1,2))
 valid3=ct.reshape(valid,(8,2,2))
 valida=ct.extract(valid3,(0,0,0),(8,1,2));validb=ct.extract(valid3,(0,1,0),(8,1,2))
 direction=((ct.arange(8,dtype=ct.int32)[:,None,None]*2+ct.arange(2,dtype=ct.int32)[None,None,:])&8)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=16, stride=1
 keys3=ct.reshape(keys,(16,2,1))
 keysa=ct.extract(keys3,(0,0,0),(16,1,1));keysb=ct.extract(keys3,(0,1,0),(16,1,1))
 ids3=ct.reshape(ids,(16,2,1))
 idsa=ct.extract(ids3,(0,0,0),(16,1,1));idsb=ct.extract(ids3,(0,1,0),(16,1,1))
 valid3=ct.reshape(valid,(16,2,1))
 valida=ct.extract(valid3,(0,0,0),(16,1,1));validb=ct.extract(valid3,(0,1,0),(16,1,1))
 direction=((ct.arange(16,dtype=ct.int32)[:,None,None]*1+ct.arange(1,dtype=ct.int32)[None,None,:])&8)!=0
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=None, stride=16
 keys3=ct.reshape(keys,(1,2,16))
 keysa=ct.extract(keys3,(0,0,0),(1,1,16));keysb=ct.extract(keys3,(0,1,0),(1,1,16))
 ids3=ct.reshape(ids,(1,2,16))
 idsa=ct.extract(ids3,(0,0,0),(1,1,16));idsb=ct.extract(ids3,(0,1,0),(1,1,16))
 valid3=ct.reshape(valid,(1,2,16))
 valida=ct.extract(valid3,(0,0,0),(1,1,16));validb=ct.extract(valid3,(0,1,0),(1,1,16))
 direction=False
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=None, stride=8
 keys3=ct.reshape(keys,(2,2,8))
 keysa=ct.extract(keys3,(0,0,0),(2,1,8));keysb=ct.extract(keys3,(0,1,0),(2,1,8))
 ids3=ct.reshape(ids,(2,2,8))
 idsa=ct.extract(ids3,(0,0,0),(2,1,8));idsb=ct.extract(ids3,(0,1,0),(2,1,8))
 valid3=ct.reshape(valid,(2,2,8))
 valida=ct.extract(valid3,(0,0,0),(2,1,8));validb=ct.extract(valid3,(0,1,0),(2,1,8))
 direction=False
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=None, stride=4
 keys3=ct.reshape(keys,(4,2,4))
 keysa=ct.extract(keys3,(0,0,0),(4,1,4));keysb=ct.extract(keys3,(0,1,0),(4,1,4))
 ids3=ct.reshape(ids,(4,2,4))
 idsa=ct.extract(ids3,(0,0,0),(4,1,4));idsb=ct.extract(ids3,(0,1,0),(4,1,4))
 valid3=ct.reshape(valid,(4,2,4))
 valida=ct.extract(valid3,(0,0,0),(4,1,4));validb=ct.extract(valid3,(0,1,0),(4,1,4))
 direction=False
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=None, stride=2
 keys3=ct.reshape(keys,(8,2,2))
 keysa=ct.extract(keys3,(0,0,0),(8,1,2));keysb=ct.extract(keys3,(0,1,0),(8,1,2))
 ids3=ct.reshape(ids,(8,2,2))
 idsa=ct.extract(ids3,(0,0,0),(8,1,2));idsb=ct.extract(ids3,(0,1,0),(8,1,2))
 valid3=ct.reshape(valid,(8,2,2))
 valida=ct.extract(valid3,(0,0,0),(8,1,2));validb=ct.extract(valid3,(0,1,0),(8,1,2))
 direction=False
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 # Network size=None, stride=1
 keys3=ct.reshape(keys,(16,2,1))
 keysa=ct.extract(keys3,(0,0,0),(16,1,1));keysb=ct.extract(keys3,(0,1,0),(16,1,1))
 ids3=ct.reshape(ids,(16,2,1))
 idsa=ct.extract(ids3,(0,0,0),(16,1,1));idsb=ct.extract(ids3,(0,1,0),(16,1,1))
 valid3=ct.reshape(valid,(16,2,1))
 valida=ct.extract(valid3,(0,0,0),(16,1,1));validb=ct.extract(valid3,(0,1,0),(16,1,1))
 direction=False
 swap=(((keysa>keysb)&valida)|(~validb))==direction
 keys=ct.reshape(ct.cat((ct.where(swap,keysb,keysa),ct.where(swap,keysa,keysb)),axis=1),(32,))
 ids=ct.reshape(ct.cat((ct.where(swap,idsb,idsa),ct.where(swap,idsa,idsb)),axis=1),(32,))
 valid=ct.reshape(ct.cat((ct.where(swap,validb,valida),ct.where(swap,valida,validb)),axis=1),(32,))
 ct.scatter(Y,(row,lane),ids.astype(ct.int64),check_bounds=True)
