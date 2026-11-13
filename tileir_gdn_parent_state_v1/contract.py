def layout(shape,stride,offset,item,storage):
 assert len(shape)==4 and len(stride)==4 and all(type(n)is int and n>0 for n in shape)
 n,h,k,v=shape;inner=h*k*v
 assert n<=512 and inner<=524288 and item==4 and stride[1:]==(k*v,v,1) and stride[0]>=inner and offset>=0
 end=(offset+(n-1)*stride[0]+inner)*item
 assert end<=storage<=32*1024**3 and (end-offset*item)<=1024**3
 return inner
def index_values(values,rows,scatter=False):
 assert 1<=len(values)<=32 and all(type(x)is int and -rows<=x<rows for x in values)
 normalized=[x%rows for x in values]
 assert not scatter or len(set(normalized))==len(normalized),'Duplicate scatter index outside supported contract'
 return normalized
