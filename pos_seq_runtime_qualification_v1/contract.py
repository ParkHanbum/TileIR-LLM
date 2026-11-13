"""CPU full-storage oracle; never initializes CUDA."""
import json,hashlib
from pathlib import Path
from tileir_vllm_pos_seq_v1.api import reference_checked
NAMES=('idx_mapping','query_start_loc','num_computed_tokens','pos','seq_lens')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load_record(item):
 import torch
 assert sha(item['path'])==item['sha256'] and sha(item['storage_path'])==item['storage_sha256']
 raw=torch.load(item['storage_path'],map_location='cpu',weights_only=True);logical=torch.load(item['path'],map_location='cpu',weights_only=True)
 m=item['metadata'];assert raw.dtype==torch.uint8 and raw.numel()==m['storage_bytes']
 assert hashlib.sha256(raw.numpy().tobytes()).hexdigest()==item['storage_byte_sha256']
 dtype=getattr(torch,m['dtype'].split('.')[-1]);view=torch.empty(0,dtype=dtype).set_(raw.untyped_storage(),m['storage_offset'],m['shape'],m['stride'])
 assert torch.equal(view,logical) and hashlib.sha256(logical.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==item['logical_sha256']
 return raw,view

def verify_case(s):
 import torch
 assert s['computed_placement']['has_uva_buffer'] is False,'Observed device-only branch required; no implicit UVA copy control'
 assert set(s['pre'])==set(s['post'])==set(NAMES) and s['return_kind']=='None'
 assert not any(v['storage_overlap'] for v in s['pre_aliases'].values())
 before={k:load_record(s['pre'][k]) for k in NAMES};after={k:load_record(s['post'][k]) for k in NAMES}
 args=[before[k][1].tolist() for k in NAMES];expected=reference_checked(*args)
 for k,values in zip(('pos','seq_lens'),expected):
  raw=before[k][0].clone();m=s['pre'][k]['metadata'];v=torch.empty(0,dtype=before[k][1].dtype).set_(raw.untyped_storage(),m['storage_offset'],m['shape'],m['stride']);v.copy_(torch.tensor(values,dtype=v.dtype));assert torch.equal(raw,after[k][0]),('full output storage mismatch',k)
 for k in NAMES[:3]:assert torch.equal(before[k][0],after[k][0]),('input changed',k)
 n=len(args[0]);start,end=args[1][0],args[1][n]
 return dict(phase=s['phase'],full_storage_oracle_pass=True,requests=n,live_position_range=[start,end],untouched_position_elements=len(args[3])-(end-start),seq_padding_elements=len(args[4])-n,computed_placement='regular device StagedWriteTensor; no _uva_buf')
