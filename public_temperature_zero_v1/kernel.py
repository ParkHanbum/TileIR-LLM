import cuda.tile as ct
@ct.kernel
def gather_raw(X,IDs,Y,n:ct.Constant[int],width:ct.Constant[int]):
 r=ct.arange(width,dtype=ct.int32)
 ids=ct.gather(IDs,r,check_bounds=True,padding_value=0)
 bits=ct.gather(X,ids,check_bounds=True,padding_value=0)
 ct.scatter(Y,r,bits,check_bounds=True)
