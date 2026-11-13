import struct
from tileir_staged_typed_v2.contract import validate
def expected(raw,shape,stride,dtype,data):
 intervals=validate(shape,stride,dtype,*data);code={'torch.int32':'i','torch.int64':'q','torch.float32':'f'}[dtype];size=struct.calcsize(code)
 assert len(raw)==size*shape[0]*(shape[1]if len(shape)==2 else 1)
 out=bytearray(raw);prior=0
 for (lo,hi),end in zip(intervals,data[3]):
  for i,value in enumerate(data[2][prior:end]):struct.pack_into('<'+code,out,(lo+i)*size,value)
  prior=end
 return bytes(out)
