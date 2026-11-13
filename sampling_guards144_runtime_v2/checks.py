from collections import Counter
LIMITS={'greedy.nan':16,'greedy.posinf':16,'greedy.rows':16,'ranks.nan':16,'ranks.posinf':16,'ranks.gather':16}
class Checks:
 def __init__(self,controller):
  self.controller=controller;self.active=False;self.busy=False;self.shadow_pass=False;self.rows=[];self.temperatures=[]
 def start(self,mode):
  assert not self.active and self.controller.failed is None and mode in('shadow','replace')and(mode!='replace'or self.shadow_pass)
  self.mode=mode;self.counts=Counter();self.rows=[];self.temperatures=[];self.active=True
 def _enter(self,site,x):
  import torch
  assert self.active and self.controller.active and self.controller.failed is None and not self.busy
  assert site in LIMITS and self.counts[site]<16
  assert tuple(x.shape)==(32,248320)and tuple(x.stride())==(248320,1)and x.storage_offset()==0
  assert x.dtype==(torch.float32 if site.startswith('greedy')else torch.bfloat16)
  assert x.is_cuda and not x.requires_grad and not x.is_neg()and not x.is_conj()
  self.busy=True
  row=dict(site=site,index=self.counts[site]+1,mode=self.mode,stage='entered');self.rows.append(row);return row
 def _observe(self,x):
  import torch
  storage=x.untyped_storage();assert storage.nbytes()<=64*1024*1024
  meta=lambda:(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.data_ptr(),x.is_neg(),x.is_conj())
  def snapshot():return torch.empty(0,dtype=torch.uint8,device=x.device).set_(storage,0,(storage.nbytes(),),(1,)).cpu()
  return meta,meta(),snapshot,snapshot()
 def predicate(self,site,x,mode):
  import torch
  from public_sampling_guard_v1.api import classify
  row=None
  try:
   row=self._enter(site,x);assert mode=={'greedy.nan':'has_nan','greedy.posinf':'has_posinf','greedy.rows':'rows_have_finite','ranks.nan':'has_nan','ranks.posinf':'has_posinf'}[site]
   meta,desc,snapshot,before=self._observe(x);stream=torch.cuda.current_stream(x.device).cuda_stream
   native=None
   if self.mode=='shadow':
    row['stage']='native_attempt'
    native=bool(torch.isnan(x).any())if mode=='has_nan'else(bool(torch.isposinf(x).any())if mode=='has_posinf'else bool(torch.isfinite(x).any(dim=-1).all()))
    assert torch.equal(before,snapshot()),'Native predicate changed input'
   row['stage']='candidate_attempt';tile=classify(x,mode);assert type(tile)is bool and(native is None or native is tile),'Sampling predicate mismatch'
   assert meta()==desc and torch.equal(before,snapshot())and torch.cuda.current_stream(x.device).cuda_stream==stream
   row.update(stage='verified',kind='predicate',predicate=mode,native=native,tile=tile,input_owner_preserved=True,metadata_preserved=True,same_stream=True,descriptor=desc[:4])
   self.counts[site]+=1;return native if self.mode=='shadow'else tile
  except BaseException as e:
   if row is not None:row['error']=repr(e)
   self.controller.latch('sampling96',e);raise
  finally:self.busy=False
 def gather(self,x,ids):
  import torch
  from public_sampling_guard_v1.api import selected_columns_checked
  row=None
  try:
   row=self._enter('ranks.gather',x)
   assert ids.dtype==torch.int64 and tuple(ids.shape)==(32,)and ids.device==x.device
   xm,xd,xsnap,xb=self._observe(x);im,id_,isnap,ib=self._observe(ids);stream=torch.cuda.current_stream(x.device).cuda_stream
   native=x.gather(1,ids[:,None])if self.mode=='shadow'else None
   assert torch.equal(xb,xsnap())and torch.equal(ib,isnap())
   row['stage']='candidate_attempt';tile=selected_columns_checked(x,ids)
   assert tuple(tile.shape)==(32,1)and tile.dtype==x.dtype and tile.device==x.device and tile.is_contiguous()
   assert tile.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
   if native is not None:assert torch.equal(native.view(torch.uint8),tile.view(torch.uint8)),'Selected raw bits differ'
   assert xm()==xd and im()==id_ and torch.equal(xb,xsnap())and torch.equal(ib,isnap())and torch.cuda.current_stream(x.device).cuda_stream==stream
   row.update(stage='verified',kind='gather',raw_comparison=self.mode=='shadow',input_owner_preserved=True,metadata_preserved=True,same_stream=True,output_fresh=True,output_shape=[32,1],internal_checked_range_calls=1)
   self.counts['ranks.gather']+=1;return native if self.mode=='shadow'else tile
  except BaseException as e:
   if row is not None:row['error']=repr(e)
   self.controller.latch('sampling96',e);raise
  finally:self.busy=False
 def observe_temperature(self,temp,idx):
  try:
   assert self.active and self.controller.active and self.controller.failed is None and len(self.temperatures)<16
   def meta(x):return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),storage_offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),device=str(x.device),requires_grad=x.requires_grad,is_neg=x.is_neg(),is_conj=x.is_conj())
   self.temperatures.append(dict(index=len(self.temperatures)+1,temperature=meta(temp),idx=meta(idx),values_read=False,original_temperature_predicate_retained=True))
  except BaseException as e:self.controller.latch('sampling96-temperature-metadata',e);raise
 def finish(self):
  assert self.active and not self.busy and self.controller.failed is None and dict(self.counts)==LIMITS and len(self.rows)==96
  assert all(r['stage']=='verified'for r in self.rows)and len(self.temperatures)==16
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(mode=self.mode,calls=96,counts=dict(self.counts),rows=self.rows,temperature_metadata=self.temperatures,original_predicates=80 if self.mode=='shadow'else 0,original_gathers=16 if self.mode=='shadow'else 0,tile_predicates=80,tile_gathers=16,additional_internal_range_predicates=16,shadow_pass=self.shadow_pass)
