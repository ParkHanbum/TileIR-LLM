# SPDX-License-Identifier: Apache-2.0
# Independent implementation of public post-update safety predicates.
import cuda.tile as ct
@ct.kernel
def stage_guard(idx,computed,hist,sampled,ns,nr,q,total,out,stage:ct.Constant[int]):
    row=ct.arange(32,dtype=ct.int32)
    state=ct.gather(idx,row,check_bounds=True)
    active=state>=0
    safe_state=ct.where(active,state,0)
    if stage==0:
        bad=active & (state>=32)
    elif stage==1:
        left=ct.expand_dims(state,1);right=ct.expand_dims(state,0)
        ri=ct.expand_dims(row,1);rj=ct.expand_dims(row,0)
        duplicate=(left>=0)&(right>=0)&(left==right)&(ri!=rj)
        bad=ct.sum(ct.astype(duplicate,ct.int32),1)>0
    elif stage==2:
        n=ct.astype(ct.gather(ns,row,check_bounds=True),ct.int64)
        bad=active & ((n<0)|(n>1))
    elif stage==3:
        n=ct.astype(ct.gather(ns,row,check_bounds=True),ct.int64)
        old=ct.astype(ct.gather(total,safe_state,check_bounds=True),ct.int64)
        bad=active & ((old<0)|(old+n>128))
    elif stage==4:
        n=ct.gather(ns,row,check_bounds=True)
        token=ct.gather(sampled,row,check_bounds=True)
        accepted=active & (n>0)
        bad=accepted & ((token<0)|(token>=248320))
    elif stage==5:
        n=ct.gather(ns,row,check_bounds=True)
        token=ct.gather(sampled,row,check_bounds=True)
        accepted=active & (n>0)
        # Stage4 must pass before this dependent access. Inactive safe loads
        # only touch known-valid element0; they cannot contribute a failure.
        address=ct.where(accepted,state*248320+token,0)
        count=ct.astype(ct.gather(hist,address,check_bounds=True),ct.int64)
        bad=accepted & (count>=2147483647)
    elif stage==6:
        left=ct.astype(ct.gather(q,row,check_bounds=True),ct.int64)
        right=ct.astype(ct.gather(q,row+1,check_bounds=True),ct.int64)
        delta=right-left
        bad=active & ((delta < -2147483648)|(delta>2147483647))
    elif stage==7:
        left=ct.astype(ct.gather(q,row,check_bounds=True),ct.int64)
        right=ct.astype(ct.gather(q,row+1,check_bounds=True),ct.int64)
        rejected=ct.astype(ct.gather(nr,row,check_bounds=True),ct.int64)
        delta=right-left-rejected
        bad=active & ((delta < -2147483648)|(delta>2147483647))
    else:
        left=ct.astype(ct.gather(q,row,check_bounds=True),ct.int64)
        right=ct.astype(ct.gather(q,row+1,check_bounds=True),ct.int64)
        rejected=ct.astype(ct.gather(nr,row,check_bounds=True),ct.int64)
        old=ct.astype(ct.gather(computed,safe_state,check_bounds=True),ct.int64)
        updated=old+right-left-rejected
        bad=active & ((updated < -2147483648)|(updated>2147483647))
    ct.scatter(out,0,ct.sum(ct.astype(bad,ct.int32),0),check_bounds=True)
