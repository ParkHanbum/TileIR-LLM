# SPDX-License-Identifier: Apache-2.0
# Public vLLM/FLA cumsum.py semantics independently expressed in cuTile.
# Source attribution and original MIT notice retained in README.md and LICENSE.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def scalar_cumsum(G,O,Cu,Indices):
    chunk=ct.bid(0);head=ct.bid(1)
    seq=ct.gather(Indices,(chunk,0),check_bounds=False).astype(ct.int32)
    local=ct.gather(Indices,(chunk,1),check_bounds=False).astype(ct.int32)
    begin=ct.gather(Cu,seq,check_bounds=False).astype(ct.int32)
    end=ct.gather(Cu,seq+1,check_bounds=False).astype(ct.int32)
    token=begin+local*64+ct.arange(64,dtype=ct.int32)
    active=token<end
    x=ct.gather(G,(0,token,head),mask=active,padding_value=0,check_bounds=False)
    y=ct.cumsum(x,axis=0,rounding_mode=RM.RN,flush_to_zero=False)
    ct.scatter(O,(0,token,head),y,mask=active,check_bounds=False)
