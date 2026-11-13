"""Chunked exact BF16 metrics; torch is passed in, no backend import here."""
import hashlib
CHUNK_BYTES=16*1024*1024

def chunks(t):
 assert t.is_contiguous()
 flat=t.detach().reshape(-1);step=CHUNK_BYTES//t.element_size()
 for start in range(0,flat.numel(),step):yield start,flat[start:start+step].cpu().contiguous()

def digest(t,torch):
 h=hashlib.sha256();finite=True;count=0
 for _,c in chunks(t):
  h.update(c.view(torch.uint8).numpy().tobytes());finite=finite and bool(torch.isfinite(c).all());count+=c.numel()
 return dict(sha256=h.hexdigest(),finite=finite,elements=count,bytes=count*t.element_size())

def compare(native,tile,torch):
 assert native.shape==tile.shape and native.dtype==tile.dtype==torch.bfloat16 and native.ndim==2
 nh=hashlib.sha256();th=hashlib.sha256();raw=strict=nf=tf=0;maximum=0.;first=[];count=0
 for (a0,a),(b0,b) in zip(chunks(native),chunks(tile),strict=True):
  assert a0==b0 and a.shape==b.shape
  nh.update(a.view(torch.uint8).numpy().tobytes());th.update(b.view(torch.uint8).numpy().tobytes())
  af=a.float();bf=b.float();finite=torch.isfinite(af)&torch.isfinite(bf);error=(af-bf).abs();mask=a.view(torch.int16)!=b.view(torch.int16)
  raw+=int(mask.sum());strict+=int(((error>1e-5+1e-5*af.abs())|~finite).sum());nf+=int((~torch.isfinite(af)).sum());tf+=int((~torch.isfinite(bf)).sum());count+=a.numel()
  if bool(finite.any()):maximum=max(maximum,float(error[finite].max()))
  if len(first)<16:
   for i in mask.nonzero().flatten()[:16-len(first)].tolist():
    absolute=a0+i;first.append(dict(index=[absolute//native.shape[1],absolute%native.shape[1]],native_bits=int(a[i].view(torch.int16))&65535,tile_bits=int(b[i].view(torch.int16))&65535))
 return dict(raw_equal=raw==0,raw_mismatches=raw,strict_failures=strict,reference_nonfinite=nf,candidate_nonfinite=tf,elements=count,max_abs_finite=maximum,native_sha256=nh.hexdigest(),tile_sha256=th.hexdigest(),first_differences=first,strict_tolerance='1e-5+1e-5*abs(native)')

def passing(c):return c['raw_equal'] and c['raw_mismatches']==c['strict_failures']==c['reference_nonfinite']==c['candidate_nonfinite']==0
