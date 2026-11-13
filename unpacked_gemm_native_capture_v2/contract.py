"""Native-only five-contract capture invariants. Standard library only."""
import hashlib,json,re
from pathlib import Path
MODEL=Path('/home/n/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
GROUPS={'gdn_out':('linear_attn.out_proj',24,2560,4096),
        'attn_out':('self_attn.o_proj',8,2560,4096),
        'down':('mlp.down_proj',32,2560,9216),
        'head':('lm_head',1,248320,2560)}
REP={'gdn_out':'layers.0.linear_attn.out_proj','attn_out':'layers.3.self_attn.o_proj','down':'layers.0.mlp.down_proj','head':'lm_head'}
WEIGHT_KEYS={'gdn_out':'model.language_model.layers.0.linear_attn.out_proj.weight','attn_out':'model.language_model.layers.3.self_attn.o_proj.weight','down':'model.language_model.layers.0.mlp.down_proj.weight','head':'model.language_model.embed_tokens.weight'}
CHUNK_BYTES=16*1024*1024;LOGICAL_CAP=192*1024*1024;FILE_CAP=256*1024*1024

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def kind_of(name):
    found=[k for k,(suffix,*_) in GROUPS.items() if name==suffix or name.endswith('.'+suffix)]
    assert len(found)<=1
    return found[0] if found else None

def is_representative(kind,name):
    return name==REP[kind] or name.endswith('.'+REP[kind])

def phase(kind,call_index,m,k):
    assert 0<=call_index<16 and k==GROUPS[kind][3]
    ph='prefill' if call_index==0 else 'decode'
    assert m==(32 if kind=='head' or ph=='decode' else 1696)
    return ph

def checkpoint(kind):
    index=MODEL/'model.safetensors.index.json';config=MODEL/'config.json'
    mapping=json.loads(index.read_text())['weight_map'];key=WEIGHT_KEYS[kind]
    if kind=='head':assert json.loads(config.read_text())['tie_word_embeddings'] is True
    path=MODEL/mapping[key];st=path.stat()
    return dict(key=key,path=str(path),resolved_path=str(path.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,index_sha256=sha(index),config_sha256=sha(config))

def parse_one(path,x,w,y):
    # Adapted from the existing official public logger parser; no pointer dereference.
    text=Path(path).read_text();starts=list(re.finditer(r'^I! cuBLAS',text,re.M));calls=[]
    for i,match in enumerate(starts):
        block=text[match.start():starts[i+1].start() if i+1<len(starts) else len(text)]
        if not re.search(r'function .*? cublasGemmEx\(',block):continue
        args={}
        for key,typ,val in re.findall(r'^i!\s{2}(\w+): type=(.*?); val=(.*)$',block,re.M):
            assert key not in args;args[key]=dict(type=typ,raw_value=val,value=int(val) if typ=='int' else val)
        ctx=re.search(r'^i!Process=(\d+); Thread=(\d+); GPU=(\d+); Handle=(.*?); StreamId=(.*?); MathMode=(.*)$',block,re.M);assert ctx
        calls.append(dict(arguments=args,context=dict(zip(('process','thread','gpu','handle','stream','math_mode'),ctx.groups())),block_sha256=hashlib.sha256(block.encode()).hexdigest()))
    assert len(calls)==1,'Exactly one GemmEx inside selected original apply required'
    c=calls[0];a=c['arguments'];m,k=x['shape'];n=w['shape'][0]
    expected=dict(m=n,n=m,k=k,lda=w['stride'][0],ldb=x['stride'][0],ldc=y['stride'][0])
    for key,value in expected.items():assert a[key]['value']==value,(key,a[key],value)
    for key,value in dict(transa='CUBLAS_OP_T(1)',transb='CUBLAS_OP_N(0)',Atype='CUDA_R_16BF(14)',Btype='CUDA_R_16BF(14)',Ctype='CUDA_R_16BF(14)',computeType='CUBLAS_COMPUTE_32F(68)',algo='CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)').items():assert a[key]['value']==value,(key,a[key],value)
    assert c['context']['math_mode']=='CUBLAS_DEFAULT_MATH'
    for key,meta in (('A',w),('B',x),('C',y)):
        val=a[key]['raw_value'];p=re.search(r'0x(?:0x)?([0-9a-fA-F]+)',val);assert p and int(p[1],16)==meta['data_ptr'],(key,val,meta['data_ptr'])
    for key in ('alpha','beta'):assert a[key]['raw_value'].startswith('POINTER');a[key]['value']=None
    c.update(source_linear_scalars={'alpha':1,'beta':0,'bias':None},api_scalar_values='Not exposed: pointer-only logger; source-derived values not observed scalar bytes',pointer_mode='not recorded by logger',selected_algorithm='unknown;99 is request enum',association='exactly one logged native apply, dimensions/ld/operand/output addresses matched',log_sha256=sha(path))
    return c
