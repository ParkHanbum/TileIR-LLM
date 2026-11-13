"""Ordinary Qwen packed recurrent decode candidate; not yet GPU-qualified."""
import math
def validate_metadata(m,slots,scale,norm):
 shapes={'mixed_qkv':(32,8192),'a':(32,32),'b':(32,32),'A_log':(32,),'dt_bias':(32,),'out':(32,1,32,128),'ssm_state_indices':(32,)}
 for k,shape in shapes.items():
  if m[k]['shape']!=shape:raise ValueError((k,'shape'))
 for k,v in m.items():
  dtype='torch.float32' if k in ('A_log','initial_state') else 'torch.int32' if k=='ssm_state_indices' else 'torch.bfloat16'
  if v['dtype']!=dtype or v['offset']<0 or any(type(s)is not int or s<=0 for s in v['stride']):raise ValueError((k,'dtype/stride/offset'))
 if m['initial_state']['shape'][1:]!=(32,128,128) or m['initial_state']['stride'][1:]!=(16384,128,1) or m['initial_state']['stride'][0]<524288:raise ValueError('state [slot,H,V,K] layout')
 if m['mixed_qkv']['stride'][1]!=1 or m['mixed_qkv']['stride'][0]<8192:raise ValueError('mixed stride')
 for k in ('a','b'):
  if m[k]['stride'][1]!=1 or m[k]['stride'][0]<32:raise ValueError(k+' stride')
 for k in ('A_log','dt_bias','ssm_state_indices'):
  if m[k]['stride']!=(1,):raise ValueError(k+' stride')
 if m['out']['stride']!=(4096,4096,128,1):raise ValueError('caller output contiguous')
 if type(norm)is not bool or not norm or type(scale)not in (int,float) or not math.isfinite(scale) or scale!=128**-0.5:raise ValueError('observed scale/QK norm scope')
 if len(slots)!=32 or any(type(s)is not int or s>=m['initial_state']['shape'][0] for s in slots):raise ValueError('slot bounds')
 good=[s for s in slots if s>0]
 if len(set(good))!=len(good):raise ValueError('duplicate positive slots')
 return True

def packed_decode(mixed_qkv,a,b,A_log,dt_bias,scale,initial_state,out,ssm_state_indices,use_qk_l2norm_in_kernel=True):
 import torch
 values=locals().copy();names=('mixed_qkv','a','b','A_log','dt_bias','initial_state','out','ssm_state_indices');meta={};intervals=[]
 for k in names:
  t=values[k]
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.requires_grad or t.is_conj() or t.is_neg() or t.device.type!='cuda' or t.device!=mixed_qkv.device:raise ValueError('same-device inference tensor')
  meta[k]=dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
  st=t.untyped_storage();intervals.append((st.data_ptr(),st.data_ptr()+st.nbytes()))
 for i,u in enumerate(intervals):
  for v in intervals[i+1:]:
   if max(u[0],v[0])<min(u[1],v[1]):raise ValueError('conservative separate-owner scope')
 validate_metadata(meta,ssm_state_indices.detach().cpu().tolist(),scale,use_qk_l2norm_in_kernel)
 import cuda.tile as ct
 from .kernels import core
 with torch.cuda.device(mixed_qkv.device):ct.launch(torch.cuda.current_stream(mixed_qkv.device),(32,32,4),core,(mixed_qkv,a,b,A_log,dt_bias,initial_state,out,ssm_state_indices,float(scale)))
 return out,initial_state
