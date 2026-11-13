import cuda.tile as ct
@ct.kernel
def cast_kernel(X,Y,N:ct.Constant[int],B:ct.Constant[int]):
 j=ct.bid(0)*B+ct.arange(B,dtype=ct.int32)
 x=ct.gather(X,j,mask=j<N,padding_value=0,check_bounds=True)
 y=ct.astype(x,ct.float32)
 ct.scatter(Y,j,y,mask=j<N,check_bounds=True)
