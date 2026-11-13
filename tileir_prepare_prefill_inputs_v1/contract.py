# SPDX-License-Identifier: Apache-2.0
SHAPES=((2048,),(1,32),(32,),(33,),(32,128),(32,),(32,))
STRIDES=((1,),(32,1),(1,),(1,),(128,1),(1,),(1,))
def validate_values(mapping,starts,lengths,computed):
 assert len(mapping)==32 and len(set(mapping))==32 and all(type(x)is int and 0<=x<32 for x in mapping)
 assert len(starts)==33 and all(type(x)is int and 0<=x<=2048 for x in starts) and all(a<=b for a,b in zip(starts,starts[1:]))
 assert len(lengths)==len(computed)==32 and all(type(x)is int and 0<=x<=128 for x in lengths+computed)
 for b,r in enumerate(mapping):
  if computed[r]<lengths[r]:assert computed[r]+starts[b+1]-starts[b]<=128
