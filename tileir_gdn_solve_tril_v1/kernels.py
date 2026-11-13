# SPDX-License-Identifier: Apache-2.0
# Public vLLM/FLA solve_tril.py block algebra; original MIT attribution in LICENSE.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def inverse(A,O):
    seq=ct.bid(0);head=ct.bid(1)
    i=ct.arange(16,dtype=ct.int32);lower=i[:,None]>i[None,:];eye=i[:,None]==i[None,:]
    zero=ct.full((16,16),0,dtype=ct.float32)
    d1=-ct.where(lower,ct.gather(A,(0,seq*53+0+i[:,None],head,0+i[None,:]),mask=(0+i[:,None]<53),padding_value=0,check_bounds=False),0)
    for row in range(2,16):
        a=-ct.gather(A,(0,seq*53+0+row,head,0+i),check_bounds=False)
        product=ct.mul(a[:,None],d1,rounding_mode=RM.RN,flush_to_zero=False)
        value=ct.add(a,ct.sum(product,axis=0,rounding_mode=RM.RN,flush_to_zero=False),rounding_mode=RM.RN,flush_to_zero=False)
        d1=ct.where((i==row)[:,None],value,d1)
    d1=ct.add(d1,eye.astype(ct.float32),rounding_mode=RM.RN,flush_to_zero=False)
    d2=-ct.where(lower,ct.gather(A,(0,seq*53+16+i[:,None],head,16+i[None,:]),mask=(16+i[:,None]<53),padding_value=0,check_bounds=False),0)
    for row in range(2,16):
        a=-ct.gather(A,(0,seq*53+16+row,head,16+i),check_bounds=False)
        product=ct.mul(a[:,None],d2,rounding_mode=RM.RN,flush_to_zero=False)
        value=ct.add(a,ct.sum(product,axis=0,rounding_mode=RM.RN,flush_to_zero=False),rounding_mode=RM.RN,flush_to_zero=False)
        d2=ct.where((i==row)[:,None],value,d2)
    d2=ct.add(d2,eye.astype(ct.float32),rounding_mode=RM.RN,flush_to_zero=False)
    d3=-ct.where(lower,ct.gather(A,(0,seq*53+32+i[:,None],head,32+i[None,:]),mask=(32+i[:,None]<53),padding_value=0,check_bounds=False),0)
    for row in range(2,16):
        a=-ct.gather(A,(0,seq*53+32+row,head,32+i),check_bounds=False)
        product=ct.mul(a[:,None],d3,rounding_mode=RM.RN,flush_to_zero=False)
        value=ct.add(a,ct.sum(product,axis=0,rounding_mode=RM.RN,flush_to_zero=False),rounding_mode=RM.RN,flush_to_zero=False)
        d3=ct.where((i==row)[:,None],value,d3)
    d3=ct.add(d3,eye.astype(ct.float32),rounding_mode=RM.RN,flush_to_zero=False)
    d4=-ct.where(lower,ct.gather(A,(0,seq*53+48+i[:,None],head,48+i[None,:]),mask=(48+i[:,None]<53),padding_value=0,check_bounds=False),0)
    for row in range(2,5):
        a=-ct.gather(A,(0,seq*53+48+row,head,48+i),check_bounds=False)
        product=ct.mul(a[:,None],d4,rounding_mode=RM.RN,flush_to_zero=False)
        value=ct.add(a,ct.sum(product,axis=0,rounding_mode=RM.RN,flush_to_zero=False),rounding_mode=RM.RN,flush_to_zero=False)
        d4=ct.where((i==row)[:,None],value,d4)
    d4=ct.add(d4,eye.astype(ct.float32),rounding_mode=RM.RN,flush_to_zero=False)
    a21=ct.gather(A,(0,seq*53+16+i[:,None],head,0+i[None,:]),mask=(16+i[:,None]<53),padding_value=0,check_bounds=False)
    a31=ct.gather(A,(0,seq*53+32+i[:,None],head,0+i[None,:]),mask=(32+i[:,None]<53),padding_value=0,check_bounds=False)
    a32=ct.gather(A,(0,seq*53+32+i[:,None],head,16+i[None,:]),mask=(32+i[:,None]<53),padding_value=0,check_bounds=False)
    a41=ct.gather(A,(0,seq*53+48+i[:,None],head,0+i[None,:]),mask=(48+i[:,None]<53),padding_value=0,check_bounds=False)
    a42=ct.gather(A,(0,seq*53+48+i[:,None],head,16+i[None,:]),mask=(48+i[:,None]<53),padding_value=0,check_bounds=False)
    a43=ct.gather(A,(0,seq*53+48+i[:,None],head,32+i[None,:]),mask=(48+i[:,None]<53),padding_value=0,check_bounds=False)
    x21=-ct.mma(ct.mma(d2,a21,zero),d1,zero)
    x32=-ct.mma(ct.mma(d3,a32,zero),d2,zero)
    x43=-ct.mma(ct.mma(d4,a43,zero),d3,zero)
    x31=-ct.mma(d3,(ct.mma(a31,d1,zero)+ct.mma(a32,x21,zero)),zero)
    x42=-ct.mma(d4,(ct.mma(a42,d2,zero)+ct.mma(a43,x32,zero)),zero)
    x41=-ct.mma(d4,((ct.mma(a41,d1,zero)+ct.mma(a42,x21,zero))+ct.mma(a43,x31,zero)),zero)
    full=ct.cat((ct.cat((d1,zero,zero,zero),axis=1),ct.cat((x21,d2,zero,zero),axis=1),ct.cat((x31,x32,d3,zero),axis=1),ct.cat((x41,x42,x43,d4),axis=1)),axis=0)
    j=ct.arange(64,dtype=ct.int32)
    ct.scatter(O,(0,seq*53+j[:,None],head,j[None,:]),full.astype(ct.bfloat16,rounding_mode=RM.RN),mask=j[:,None]<53,check_bounds=False)
