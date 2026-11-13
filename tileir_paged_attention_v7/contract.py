def validate_values(cuq,lens,table,capacity,page_size):
 assert len(cuq)==33 and cuq[0]==0 and len(lens)==32 and len(table)==32 and capacity>0 and page_size==528
 widths=[b-a for a,b in zip(cuq,cuq[1:])];assert widths in ([53]*32,[1]*32)
 assert all(type(n)is int and 53<=n<=64 for n in lens)
 if widths[0]==53:assert lens==[53]*32
 else:assert len(set(lens))==1 and lens[0]>=54
 for row,L in zip(table,lens):assert len(row)>=1 and all(type(b)is int and 0<=b<capacity for b in row[:(L+page_size-1)//page_size])
 return widths[0]
