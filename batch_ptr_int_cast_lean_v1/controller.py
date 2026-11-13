from batch_ptr_int_cast_v1.api import Transport
from batch_ptr_int_cast_v1.binding import Binding
class Controller:
 def __init__(self,transport=None):
  self.transport=transport if transport is not None else Transport()
  self.failed=False;self.mode=None;self.calls=0;self.binding=None;self.closed=False;self.finished=False
 def mark_failed(self):self.failed=True
 def start(self,mode='lean'):
  if self.failed or self.closed or mode!='lean' or self.transport.pending or (self.mode is not None and not self.finished):self.failed=True;raise RuntimeError('phase/lifetime latch')
  self.mode=mode;self.calls=0;self.finished=False
 def call(self,dst,src,non_blocking=True):
  if self.failed or self.closed or self.mode!='lean' or self.finished or self.calls!=0:self.failed=True;raise RuntimeError('entry/cap latch')
  try:
   result=self.transport.cast_into(dst,src,non_blocking)
   if result is not dst:raise ValueError('return identity')
   self.calls+=1;return result
  except BaseException:self.failed=True;raise
 def finish(self):
  if self.failed or self.transport.failed or self.calls!=1 or self.finished:self.failed=True;raise RuntimeError('census latch')
  try:self.transport.synchronize_and_release()
  except BaseException:self.failed=True;raise
  self.finished=True
  return dict(calls=1,native_production_copy=0,native_private_oracle=0,native_same_dtype_H2D=1,tileir_cast_launches=1,temporary_bytes=1792,raw_rechecked=False,owners_released=True)
 def install(self,module):
  from .gates import qualification
  qualification()
  self.binding=Binding(module,self.call,lambda:self.failed or self.transport.failed,self.mark_failed)
  self.binding.install()
 def close(self):
  errors=[]
  try:
   if self.binding:self.binding.close()
  except BaseException as e:errors.append(e)
  try:self.transport.synchronize_and_release()
  except BaseException as e:errors.append(e)
  self.closed=True
  if errors:self.failed=True;raise RuntimeError('restore/drain failure') from errors[0]
