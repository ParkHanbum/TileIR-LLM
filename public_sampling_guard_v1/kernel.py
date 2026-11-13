# SPDX-License-Identifier: Apache-2.0
# Independent public TileIR predicates, bit-defined IEEE classes; no vendor binary input.
import cuda.tile as ct
@ct.kernel
def classify_partial(X,P,cols:ct.Constant[int],mode:ct.Constant[int],absolute:ct.Constant[int],inf:ct.Constant[int]):
 row=ct.bid(0);group=ct.bid(1);col=group*1024+ct.arange(1024,dtype=ct.int32)
 bits=ct.gather(X,(row,col),check_bounds=True)
 if mode==0:
  selected=(bits&absolute)>inf
 elif mode==1:
  selected=bits==inf
 else:
  selected=(bits&absolute)<inf
 count=ct.sum(((col<cols)&selected).astype(ct.int32),axis=0)
 ct.scatter(P,(row,group),count,check_bounds=True)
@ct.kernel
def classify_finish(P,O,rows:ct.Constant[int],groups:ct.Constant[int],rw:ct.Constant[int],gw:ct.Constant[int],mode:ct.Constant[int]):
 r=ct.arange(rw,dtype=ct.int32);g=ct.arange(gw,dtype=ct.int32)
 counts=ct.gather(P,(r[:,None],g[None,:]),check_bounds=True,padding_value=0)
 per_row=ct.sum(ct.where(g[None,:]<groups,counts,0),axis=1)
 if mode==2:
  total=ct.sum(((r<rows)&(per_row==0)).astype(ct.int32),axis=0)
 else:
  total=ct.sum(ct.where(r<rows,per_row,0),axis=0)
 ct.scatter(O,0,total,check_bounds=True)
@ct.kernel
def gather_columns(X,IDs,Y,rows:ct.Constant[int],width:ct.Constant[int]):
 r=ct.arange(width,dtype=ct.int32)
 ids=ct.gather(IDs,r,check_bounds=True)
 value=ct.gather(X,(r,ids),check_bounds=True)
 ct.scatter(Y,(r,0),value,check_bounds=True)
