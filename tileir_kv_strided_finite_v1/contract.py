# SPDX-License-Identifier: Apache-2.0

def descriptor(shape,stride,dtype,offset,storage_bytes):
 assert tuple(shape) in ((32,4,256),(1696,4,256))
 assert tuple(stride)==(10240,256,1) and dtype=='torch.bfloat16' and offset==9216
 n=shape[0];last=offset+(n-1)*10240+3*256+255
 assert type(storage_bytes)is int and storage_bytes==n*10240*2
 assert (last+1)*2<=storage_bytes and last<2**31
 return n

def logical_address(row,head,channel):return 9216+row*10240+head*256+channel
