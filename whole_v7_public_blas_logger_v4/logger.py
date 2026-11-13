"""Public logger APIs only. No callback, interposition, binary inspection or new library load."""
import ctypes,os,re,json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.logger import PublicLogger,snapshot
R=Path(__file__).resolve().parent

def summarize(text):
 names=re.findall(r'^I! cuBLAS .*? function .*?\b(cublas\w+)\(',text,re.M)
 return {n:names.count(n) for n in sorted(set(names))}
class Window:
 def __init__(self,output,mode):
  assert mode in ('native','lean');self.mode=mode;self.base=Path(output);self.classic=None;self.lt=None;self.record={'mode':mode,'lt_supported':False};self.calls=[]
 def ltcall(self,name,*args):
  status=int(getattr(self.lt,name)(*args));self.calls.append(dict(api=name,status=status));assert status==0,(name,status)
 def __enter__(self):
  expected=json.loads((R/'LIBRARIES.json').read_text());mapped={str(Path(line.split()[-1]).resolve()) for line in Path('/proc/self/maps').read_text().splitlines() if '/libcublas' in line}
  self.record['mapped_libraries']=sorted(mapped)
  assert mapped <= {expected[k]['path'] for k in ('classic','lt')},'Unexpected mapped vendor library'
  try:
   self.classic=PublicLogger(self.base/(self.mode+'.cublas.log'))
   lt=expected['lt'];path=Path(lt['path']).resolve()
   if str(path) in mapped:
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==lt['sha256']
    self.lt=ctypes.CDLL(str(path),mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
    for name,types in [('cublasLtLoggerOpenFile',[ctypes.c_char_p]),('cublasLtLoggerSetMask',[ctypes.c_int]),('cublasLtLoggerSetLevel',[ctypes.c_int])]:
     fn=getattr(self.lt,name);fn.argtypes=types;fn.restype=ctypes.c_int
    self.lt.cublasLtCreate.argtypes=[ctypes.POINTER(ctypes.c_void_p)];self.lt.cublasLtCreate.restype=ctypes.c_int
    self.lt.cublasLtDestroy.argtypes=[ctypes.c_void_p];self.lt.cublasLtDestroy.restype=ctypes.c_int
    control=self.base/(self.mode+'.cublasLt.control.log');assert not control.exists()
    self.ltcall('cublasLtLoggerSetLevel',0);self.ltcall('cublasLtLoggerOpenFile',os.fsencode(control));self.ltcall('cublasLtLoggerSetLevel',5);self.ltcall('cublasLtLoggerSetMask',16)
    handle=ctypes.c_void_p();created=False
    try:
     self.ltcall('cublasLtCreate',ctypes.byref(handle));created=True
    finally:
     try:
      if created:self.ltcall('cublasLtDestroy',handle)
     finally:self.ltcall('cublasLtLoggerSetLevel',0)
    control_text=control.read_text() if control.exists() else ''
    self.record['lt_positive_control']=dict(log=snapshot(control),create_visible='cublasLtCreate' in control_text,destroy_visible='cublasLtDestroy' in control_text,dedicated_handle=True,model_handle_modified=False,request_scope=False)
    assert self.record['lt_positive_control']['create_visible'] and self.record['lt_positive_control']['destroy_visible'],'Lt public logger positive control unavailable'
    self.ltpath=self.base/(self.mode+'.cublasLt.log');assert not self.ltpath.exists()
    self.ltcall('cublasLtLoggerSetLevel',0);self.ltcall('cublasLtLoggerOpenFile',os.fsencode(self.ltpath));self.ltcall('cublasLtLoggerSetLevel',5);self.ltcall('cublasLtLoggerSetMask',16)
    self.record['lt_supported']=True
   else:self.record['lt_unavailable']='Expected library not mapped: not loaded for audit'
   self.classic.__enter__();return self
  except BaseException:
   self.disable();raise
 def disable(self):
  errors=[]
  if self.classic is not None:
   try:self.classic.call(0)
   except BaseException as e:errors.append(repr(e))
  if self.lt is not None:
   try:self.ltcall('cublasLtLoggerSetLevel',0)
   except BaseException as e:errors.append(repr(e))
  self.record.update(disable_errors=errors,lt_configuration_calls=self.calls)
  if errors:raise RuntimeError('Public logger disable failed: '+repr(errors))
 def __exit__(self,*exc):
  try:self.disable()
  finally:
   if self.classic is not None:
    self.record['classic']=self.classic.record;self.record['classic_log']=snapshot(self.classic.path)
    self.record['classic_api_counts']=summarize(self.classic.path.read_text() if self.classic.path.exists() else '')
   if hasattr(self,'ltpath'):
    self.record['lt_log']=snapshot(self.ltpath)
    # Preserve raw experimental format locally; do not infer APIs from heuristic lines.
   (self.base/(self.mode+'.logger.json')).write_text(json.dumps(self.record,indent=2)+'\n')
 def verify(self):
  assert snapshot(self.classic.path)==self.record['classic_log']
  if hasattr(self,'ltpath'):assert snapshot(self.ltpath)==self.record['lt_log']
  n=self.record['classic_api_counts'].get('cublasGemmEx',0)
  assert n==(2448 if self.mode=='native' else 0),('Observed GemmEx census',n)
  if self.mode=='lean':assert not self.record['classic_api_counts'],'Unexpected public cuBLAS API in lean request'
  self.record['request_gate']=True;return self.record
