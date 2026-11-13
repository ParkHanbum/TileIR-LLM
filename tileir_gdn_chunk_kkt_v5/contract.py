"""CPU-only narrow actual KKT contract, no native winner assumption."""
def descriptor(kshape,kstride,gshape,gstride,bshape,bstride,bk):
 assert tuple(kshape)==(1,1696,16,128) and tuple(kstride)==(3473408,2048,128,1)
 assert tuple(gshape)==tuple(bshape)==(1,1696,32)
 assert tuple(gstride)==tuple(bstride)==(54272,32,1)
 assert type(bk) is int and bk in (32,64,128)
 return (1,1696,32,64)
def key_head(output_head):
 assert type(output_head) is int and 0<=output_head<32
 return output_head//2
def mask(row,col):return 0<=row<53 and 0<=col<53 and row>col
