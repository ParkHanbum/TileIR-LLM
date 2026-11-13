"""CPU metrics; independent raw, finite, ULP and two named tolerance summaries."""
def metrics(a,b):
 import torch
 a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous()
 assert a.shape==b.shape and a.dtype==b.dtype==torch.bfloat16
 af,bf=a.float(),b.float();fa,fb=torch.isfinite(af),torch.isfinite(bf);finite=fa&fb;err=(af-bf).abs()
 ua=a.view(torch.int16).to(torch.int32)&65535;ub=b.view(torch.int16).to(torch.int32)&65535
 oa=torch.where(ua>=32768,65535-ua,ua+32768);ob=torch.where(ub>=32768,65535-ub,ub+32768)
 strict=(err<=1e-5+1e-5*bf.abs())&finite;source=(err<=1e-3+1.6e-2*bf.abs())&finite
 nz=finite&(bf!=0)
 return dict(elements=a.numel(),raw_equal=bool(torch.equal(ua,ub)),raw_mismatches=int((ua!=ub).sum()),candidate_nonfinite=int((~fa).sum()),reference_nonfinite=int((~fb).sum()),max_abs_finite=float(err[finite].max()) if bool(finite.any()) else None,max_relative_nonzero=float((err[nz]/bf[nz].abs()).max()) if bool(nz.any()) else None,zero_reference_nonzero_candidate=int((finite&(bf==0)&(af!=0)).sum()),max_bf16_ulp_finite=int((oa-ob).abs()[finite].max()) if bool(finite.any()) else None,strict_pass=bool(strict.all()),strict_failures=int((~strict).sum()),strict='abs<=1e-5+1e-5*abs(reference), existing diagnostic',source_reference_pass=bool(source.all()),source_reference_failures=int((~source).sum()),source_reference='abs<=1e-3+1.6e-2*abs(reference), public vLLM BF16 testing reference only; does not release raw HOLD')
