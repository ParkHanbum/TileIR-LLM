"""Fixed public sampler ID concatenation; no native fallback or global patch."""
def validate_descriptor(m,width):
 assert width in (1,5) and m['shape']==[32,width] and m['stride']==[width,1]
 assert m['offset']==0 and m['dtype']=='torch.int64' and m['bytes']==32*width*8 and not m['requires_grad']
def concat_sampled_top5(chosen,top5):
 import torch,cuda.tile as ct
 from .kernel import concat
 for t,width in ((chosen,1),(top5,5)):
  assert isinstance(t,torch.Tensor) and t.is_cuda
  validate_descriptor(dict(shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype),bytes=t.untyped_storage().nbytes(),requires_grad=t.requires_grad),width)
 assert chosen.device==top5.device
 assert chosen.untyped_storage().data_ptr()!=top5.untyped_storage().data_ptr()
 out=torch.empty((32,6),dtype=torch.int64,device=chosen.device)
 ct.launch(torch.cuda.current_stream(chosen.device),(1,),concat,(chosen,top5,out))
 return out
