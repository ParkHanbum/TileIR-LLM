"""Original public cuTile cat kernels. No pointer tricks or float arithmetic.

Tuple indexing is compile-time static. Aliased cat inputs are represented by
one source argument, not overlapping distinct array arguments. View offsets
are already included by DLPack; logical coordinates never add storage_offset.
"""
import cuda.tile as ct

@ct.kernel
def gdn32_kernel(S, D, T: ct.Constant[int], B: ct.Constant[int]):
    i = ct.astype(ct.bid(0), ct.int64) * B + ct.arange(B, dtype=ct.int64)
    batch = ct.bid(1)
    channel = i // T
    time = i % T
    for j in ct.static_iter(range(32)):
        if batch == j:
            value = ct.gather(S[j], (0, channel, time), mask=i < 8192*T, padding_value=0, check_bounds=True)
            bits = ct.bitcast(value, ct.uint16)
            ct.scatter(D, (j, channel, time), ct.bitcast(bits, ct.bfloat16), mask=i < 8192*T, check_bounds=True)

@ct.kernel
def kv_initial_kernel(S, D, T: ct.Constant[int], N: ct.Constant[int], B: ct.Constant[int]):
    i = ct.astype(ct.bid(0), ct.int64) * B + ct.arange(B, dtype=ct.int64)
    d = i % 256
    t = (i // 256) % T
    h = (i // (256*T)) % 4
    b = i // (256*T*4)
    value = ct.gather(S, (b,h,t,d), mask=i<N, padding_value=0, check_bounds=True)
    bits = ct.bitcast(value, ct.uint16)
    ct.scatter(D, (b,h,t,d), ct.bitcast(bits, ct.bfloat16), mask=i<N, check_bounds=True)

@ct.kernel
def kv_append_kernel(A, C, D, L: ct.Constant[int], N: ct.Constant[int], B: ct.Constant[int]):
    i = ct.astype(ct.bid(0), ct.int64) * B + ct.arange(B, dtype=ct.int64)
    d = i % 256
    t = (i // 256) % (L+1)
    h = (i // (256*(L+1))) % 4
    b = i // (256*(L+1)*4)
    a = ct.gather(A, (b,h,t,d), mask=(i<N)&(t<L), padding_value=0, check_bounds=True)
    c = ct.gather(C, (b,h,t-L,d), mask=(i<N)&(t>=L), padding_value=0, check_bounds=True)
    bits = ct.where(t<L, ct.bitcast(a,ct.uint16), ct.bitcast(c,ct.uint16))
    ct.scatter(D, (b,h,t,d), ct.bitcast(bits,ct.bfloat16), mask=i<N, check_bounds=True)

@ct.kernel
def rope_kernel(S, D, T: ct.Constant[int], N: ct.Constant[int], B: ct.Constant[int]):
    i = ct.astype(ct.bid(0),ct.int64)*B + ct.arange(B,dtype=ct.int64)
    k = i % 64
    t = (i // 64) % T
    b = i // (64*T)
    value = ct.gather(S,(b,t,k%32),mask=i<N,padding_value=0,check_bounds=True)
    bits = ct.bitcast(value,ct.uint32)
    ct.scatter(D,(b,t,k),ct.bitcast(bits,ct.float32),mask=i<N,check_bounds=True)

@ct.kernel
def initial_positions_kernel(S, D, N: ct.Constant[int], B: ct.Constant[int]):
    i = ct.astype(ct.bid(0),ct.int64)*B + ct.arange(B,dtype=ct.int64)
    t = i % 53
    b = (i // 53) % 32
    p = i // (53*32)
    value = ct.gather(S,(0,b,t),mask=i<N,padding_value=0,check_bounds=True)
    ct.scatter(D,(p,b,t),value,mask=i<N,check_bounds=True)

@ct.kernel
def positions_kernel(A, C, D, L: ct.Constant[int], N: ct.Constant[int], B: ct.Constant[int]):
    i = ct.astype(ct.bid(0),ct.int64)*B + ct.arange(B,dtype=ct.int64)
    t = i % (L+1)
    b = (i // (L+1)) % 32
    p = i // ((L+1)*32)
    a = ct.gather(A,(p,b,t),mask=(i<N)&(t<L),padding_value=0,check_bounds=True)
    c = ct.gather(C,(p,b,t-L),mask=(i<N)&(t>=L),padding_value=0,check_bounds=True)
    value = ct.where(t<L,a,c)
    ct.scatter(D,(p,b,t),value,mask=i<N,check_bounds=True)
