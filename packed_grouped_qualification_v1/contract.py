"""CPU manifest utilities. Does not import Torch or initialize CUDA."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent.parent

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_pins(path):
    obj=json.loads(Path(path).read_text());pins=obj.get('files',obj)
    for name,digest in pins.items():
        p=Path(name);p=p if p.is_absolute() else P/p
        if sha(p)!=digest:raise ValueError('Source pin mismatch: '+str(p))
    return pins
def segments(sample,n):
    cursor=0;names=[];widths=[]
    for s in sample['packing']:
        start,end=s['start'],s['end'];label=s['label']
        if type(start) is not int or type(end) is not int or start!=cursor or end<=start:raise ValueError('Bad segment boundary')
        if not isinstance(label,str) or not label or label in names:raise ValueError('Bad segment label')
        names.append(label);widths.append(end-start);cursor=end
    if not widths or cursor!=n:raise ValueError('Segments do not cover weight rows')
    return tuple(widths),tuple(names)
def artifact(base,entry):
    p=Path(entry['path']);p=p if p.is_absolute() else Path(base)/p
    if sha(p)!=entry['sha256']:raise ValueError('Artifact hash mismatch: '+str(p))
    return p
def layout(meta,shape):
    # Capture uses original_gpu metadata, never confuse CPU clone strides.
    if 'original_gpu' in meta:meta=meta['original_gpu']
    if list(meta['shape'])!=list(shape):raise ValueError('Capture shape mismatch')
    stride=tuple(meta['stride']);offset=meta.get('storage_offset',0)
    if len(shape)!=2 or len(stride)!=2 or stride[1]!=1 or stride[0]<shape[1] or offset<0:raise ValueError('Unsupported NT layout')
    return stride,offset

TARGETS={'qkvz':(2048,2048,4096,4096),'ba':(32,32),'attentionqkv':(8192,1024,1024),'gate_up':(9216,9216)}
def check_manifest(m):
    if m.get('schema')!='vllm-packed-boundary-v1' or m.get('complete') is not True:raise ValueError('Actual completed vLLM capture required')
    samples=m['samples'];expected={(k,p) for k in TARGETS for p in ('prefill','decode')}
    if len(samples)!=8 or {(s['kind'],s['phase']) for s in samples}!=expected:raise ValueError('Exactly eight target cases required')
    if len(m['weights'])!=4:raise ValueError('Expected four captured weights')
    for s in samples:
        w=m['weights'][s['weight_key']];n=sum(TARGETS[s['kind']]);widths,_=segments(s,n)
        if widths!=TARGETS[s['kind']]:raise ValueError('Packing mismatch')
        if s.get('x_post_logical_sha256')!=s['x']['logical_sha256']:raise ValueError('Capture input changed')
        if w.get('final_logical_sha256')!=w['logical_sha256']:raise ValueError('Capture weight changed')
        rows=1696 if s['phase']=='prefill' else 32
        if s['x']['metadata']['shape']!=[rows,2560] or w['metadata']['shape']!=[n,2560] or s['y']['metadata']['shape']!=[rows,n]:raise ValueError('Capture target shape mismatch')
    return True
