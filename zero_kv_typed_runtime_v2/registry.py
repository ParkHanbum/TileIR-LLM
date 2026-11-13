"""Public-owner-only geometry. No address-to-pointer/Tensor construction, no zeroing."""
from dataclasses import dataclass,field
CAP=256*1024**2
class Hold(RuntimeError):pass
def require(ok,code):
 if not ok:raise Hold(code)
@dataclass(repr=False)
class Allocation:
 owner:object=field(repr=False)
 storage:object=field(repr=False)
 device:str
 base:int=field(repr=False)
 size:int
 def check(self):
  t=self.owner;s=t.untyped_storage()
  require(str(t.device)==self.device and s.data_ptr()==self.base and s.nbytes()==self.size,'stale_allocation')
@dataclass(repr=False)
class Segment:
 allocation:Allocation=field(repr=False)
 view:object=field(repr=False)
 offset:int
 stride:int
 page:int
 blocks:int
 address:int=field(repr=False)
 snapshot:tuple=field(repr=False)
def signature(t):
 return (id(t),str(t.device),t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.storage_offset(),t.element_size(),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())
class Registry:
 def __init__(self,device):self.device=str(device);self.allocations={};self.segments={};self.strong_owners=[];self.owner_snapshots=[]
 def add_view(self,view,num_blocks):
  require(str(view.device)==self.device and self.device.startswith('cuda'),'device')
  require(not view.is_neg() and not view.is_conj() and not view.requires_grad,'view_bits')
  sh=tuple(view.shape);st=tuple(view.stride());el=view.element_size();storage=view.untyped_storage();base=storage.data_ptr();size=storage.nbytes();offset=view.storage_offset()*el;dp=view.data_ptr()
  require(1<=num_blocks<=4096 and len(sh)>=2 and len(sh)==len(st) and all(type(x)is int and x>0 for x in sh) and all(type(x)is int and x>=0 for x in st),'shape')
  require(el in (1,2,4,8) and 0<=offset<size<=32*1024**3 and dp==base+offset,'storage_origin')
  require(offset+el+sum((n-1)*s*el for n,s in zip(sh,st))<=size,'view_extent')
  require(sh[0]%num_blocks==0,'logical_blocks');ratio=sh[0]//num_blocks;bs=st[0]*el
  require(ratio>0 and bs>0 and bs%4==0,'stride_alignment')
  key=(self.device,base,size)
  for otherkey,a in self.allocations.items():
   require(otherkey==key or base+size<=a.base or a.base+a.size<=base,'ambiguous_allocation_overlap')
  alloc=self.allocations.setdefault(key,Allocation(view,storage,self.device,base,size));self.strong_owners.append(view);self.owner_snapshots.append(signature(view))
  outer=[i for i in range(1,len(sh)) if st[i]*el>bs];inner=[i for i in range(1,len(sh)) if i not in outer]
  page=el+sum((sh[i]-1)*st[i]*el for i in inner);stride=bs*ratio
  require(page>0 and page%4==0,'page_alignment')
  from itertools import product
  for coord in product(*(range(sh[i])for i in outer)):
   off=sum(v*st[i]*el for v,i in zip(coord,outer))
   for virtual in range(ratio):
    start=dp+off+virtual*bs;rel=start-base
    require(start%4==0,'segment_alignment')
    if start in self.segments:
     old=self.segments[start]
     require(old.allocation is alloc and old.stride==stride and old.blocks==num_blocks,'dedup_domain_mismatch')
     page_merged=max(old.page,page)
     require(rel+(num_blocks-1)*stride+page_merged<=size,'merged_extent')
     old.page=page_merged
    else:
     require(rel>=0 and rel+(num_blocks-1)*stride+page<=size,'segment_extent')
     self.segments[start]=Segment(alloc,view,rel,stride,page,num_blocks,start,signature(view))
    require(len(self.segments)<=256,'segment_count_cap')
 def check(self):
  require(all(signature(t)==snap for t,snap in zip(self.strong_owners,self.owner_snapshots)),'stale_owner_view')
  for a in self.allocations.values():a.check()
  for s in self.segments.values():require(signature(s.view)==s.snapshot,'stale_view')
 def compare(self,addresses,strides_words,pages_words,block_ids):
  self.check();require(len(addresses)==len(strides_words)==len(pages_words)==len(self.segments),'table_count')
  # Dict insertion order matches public constructor iteration and address-only dedup.
  segments=list(self.segments.values())
  for addr,sw,pw,s in zip(addresses,strides_words,pages_words,segments):
   require(type(addr)is int and type(sw)is int and type(pw)is int,'table_dtype')
   require(addr==s.address and sw*4==s.stride and pw*4==s.page,'native_table_mismatch')
  require(isinstance(block_ids,list) and len(block_ids)<=4096 and all(type(i)is int for i in block_ids),'block_ids')
  spans={}
  for s in segments:
   require(all(0<=i<s.blocks for i in block_ids),'block_id_bounds')
   key=(s.allocation.device,s.allocation.base,s.allocation.size)
   spans.setdefault(key,[]).extend((s.offset+i*s.stride,s.offset+i*s.stride+s.page)for i in block_ids)
  total=0
  for intervals in spans.values():
   end=-1
   for lo,hi in sorted(intervals):total+=max(0,hi-max(lo,end));end=max(end,hi)
  # Addresses and block values are never returned or serialized.
  result=dict(table_equal=True,segments=len(segments),allocations=len(self.allocations),selected_union_bytes=total,preimage_cap=CAP,within_budget=total<=CAP,block_count=len(block_ids),unique_block_count=len(set(block_ids)),registry_only=True,owner_bytes_written=0)
  return result

def collect_runner(runner,attention_spec_type,tensor_type):
 """Mirror public constructor over retained runner lists, never consume its generator."""
 reg=Registry(runner.device);num_blocks=runner.kv_cache_config.num_blocks
 for groups in runner.attn_groups:
  for group in groups:
   spec=group.kv_cache_spec
   if not isinstance(spec,attention_spec_type):continue
   if group.kv_cache_group_id>=len(runner.kernel_block_sizes):continue
   kernel_bs=runner.kernel_block_sizes[group.kv_cache_group_id];require(spec.block_size%kernel_bs==0,'kernel_block_ratio')
   for name in group.layer_names:
    kv=runner.compilation_config.static_forward_context[name].kv_cache
    if not isinstance(kv,tensor_type):continue
    if group.kernel_block_stride:kv=group.map_kv_cache_to_manager_blocks(kv,num_blocks)
    if kv.device.type!=runner.device.type:continue
    reg.add_view(kv,num_blocks)
 return reg
