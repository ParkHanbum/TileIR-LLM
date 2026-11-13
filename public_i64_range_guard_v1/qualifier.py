"""Bounded callback, stop on first unexpected failure; no in-process retry."""
def qualify(torch,record):
 from .api import all_in_range_i64
 from .cases import cases
 from .contract import reference
 def snapshot(t):return t.view(torch.uint8).cpu().numpy().tobytes()
 original=torch.cuda.current_stream().cuda_stream
 streams=[torch.cuda.current_stream(),torch.cuda.Stream()]
 for si,stream in enumerate(streams):
  with torch.cuda.stream(stream):
   for ci,spec in enumerate(cases()):
    row=dict(stream=si,case=ci,spec=spec,stage='input_setup',passed=False);record(row)
    v=spec['values'];off=spec['offset'];owner=torch.tensor([77]*off+v+[88,99],device='cuda',dtype=torch.int64)
    x=owner[off:off+len(v)].reshape(spec['shape']);before=snapshot(owner);expected=reference(v,spec['lower'],spec['upper']);start=stream.cuda_stream
    row['stage']='native';record(row)
    native=bool(((x>=spec['lower'])&(x<spec['upper'])).all())
    assert snapshot(owner)==before
    row['stage']='candidate_attempt';row['native']=native;record(row)
    got=all_in_range_i64(x,spec['lower'],spec['upper'])
    row['stage']='candidate_return';record(row)
    assert type(got)is bool and got is native and got is expected
    assert snapshot(owner)==before and torch.cuda.current_stream().cuda_stream==start
    row.update(stage='verified',passed=True,candidate=got,readonly_checks=2,stream_unchanged=True);record(row)
 assert torch.cuda.current_stream().cuda_stream==original
 torch.cuda.synchronize()
 return dict(complete=True,final_synchronize=True,current_stream_restored=True,model_calls=0,timing=False)
