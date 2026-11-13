"""Public module-method native observer. No replacement, weight dump or extra GEMM."""
from pathlib import Path
from collections import Counter
import functools,hashlib,inspect,json,types
from contract import GROUPS,REP,CHUNK_BYTES,LOGICAL_CAP,FILE_CAP,kind_of,is_representative,phase,checkpoint,parse_one,sha
ATTR='_unpacked_gemm_native_capture_v1'

def version(t):
    try:return t._version
    except RuntimeError as e:
        if 'Inference tensors do not track version counter' in str(e):return 'unavailable_inference_tensor'
        raise

def metadata(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),device=str(t.device),storage_nbytes=t.untyped_storage().nbytes(),data_ptr=t.data_ptr(),version=version(t))

def digest(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def identity(fn):
    fn=getattr(fn,'__func__',fn);path=inspect.getsourcefile(fn)
    return dict(module=fn.__module__,qualname=fn.__qualname__,source_file=path,source_sha256=sha(path),first_line=inspect.getsourcelines(fn)[1])

class Observer:
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(exist_ok=False);self.saved=[];self.modules=[];self.counts={};self.rows=[];self.samples=[];self.weights={};self.live_weights={};self.bytes=0;self.pending=False;self.closed=False
    def tensor(self,t,name):
        import torch
        assert t.dtype==torch.bfloat16 and t.ndim==2 and t.device.type=='cuda'
        amount=t.numel()*t.element_size();assert self.bytes+amount<=LOGICAL_CAP
        c=t.detach().cpu().contiguous();assert bool(torch.isfinite(c).all());path=self.out/(name+'.pt');assert not path.exists();torch.save(c,path);self.bytes+=amount
        return dict(path=str(path),sha256=sha(path),logical_sha256=digest(c),metadata=metadata(t),saved_shape=list(c.shape),saved_stride=list(c.stride()))
    def weight_link(self,kind,w):
        import torch
        from safetensors import safe_open
        source=checkpoint(kind);n,k=w.shape;step=max(1,CHUNK_BYTES//(k*2));live_hash=hashlib.sha256();disk_hash=hashlib.sha256();chunks=0
        assert w.dtype==torch.bfloat16 and w.stride()==(k,1) and w.storage_offset()==0
        with safe_open(source['path'],framework='pt',device='cpu') as f:
            view=f.get_slice(source['key']);assert view.get_shape()==[n,k]
            for start in range(0,n,step):
                end=min(n,start+step);cpu=view[start:end,:].contiguous();gpu_copy=w[start:end,:].detach().cpu().contiguous();assert cpu.dtype==gpu_copy.dtype==torch.bfloat16
                assert bool(torch.isfinite(cpu).all()) and torch.equal(cpu.view(torch.uint8),gpu_copy.view(torch.uint8)),'Live weight/checkpoint bytes differ'
                live_hash.update(gpu_copy.view(torch.uint8).numpy().tobytes());disk_hash.update(cpu.view(torch.uint8).numpy().tobytes());chunks+=1
        st=Path(source['path']).stat();assert (st.st_size,st.st_mtime_ns)==(source['size'],source['mtime_ns'])
        self.live_weights[kind]=w
        return dict(source=source,runtime_metadata=metadata(w),runtime_identity=id(w),logical_sha256=live_hash.hexdigest(),checkpoint_named_tensor_sha256=disk_hash.hexdigest(),full_named_tensor_byte_equal=True,chunks=chunks,chunk_bytes_upper=CHUNK_BYTES,weight_files_created=0,shard_hash_scope='Named tensor bytes verified; complete shard not rehashed')
    def verify_weights(self):
        import torch
        for kind,w in self.live_weights.items():
            rec=self.weights[kind];assert id(w)==rec['runtime_identity'];n,k=w.shape;step=max(1,CHUNK_BYTES//(k*2));h=hashlib.sha256()
            for start in range(0,n,step):h.update(w[start:min(n,start+step),:].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            rec['final_logical_sha256']=h.hexdigest();assert rec['final_logical_sha256']==rec['logical_sha256']
            st=Path(rec['source']['path']).stat();assert (st.st_size,st.st_mtime_ns)==(rec['source']['size'],rec['source']['mtime_ns'])
    def save(self):
        complete=(len(self.modules)==65 and len(self.rows)==1040 and all(v==16 for v in self.counts.values()) and len(self.samples)==8 and len(self.weights)==4 and not self.pending and all(x['passed'] and x['original_calls']==1 for x in self.rows))
        r=dict(schema='vllm-unpacked-five-contract-native-v1',complete=complete,observers_removed=self.closed,modules=self.modules,counts=self.counts,rows=self.rows,samples=self.samples,weights=self.weights,artifact_logical_bytes=self.bytes,logical_cap=LOGICAL_CAP,file_cap=FILE_CAP,source_pins=json.loads(Path(__file__).with_name('PINS.json').read_text()),limits=['Five API contracts; out/o represented separately by layer0GDN/layer3attention; two phase banks each, not all calls numerically qualified.','Native returned objects preserved; no TileIR calls or numerical qualification.','Alpha1/beta0 is public Linear-source derived; official logger exposes scalar pointers, not contents.','Weights are streamed against checkpoint named tensors, never dumped; all original files untouched.'])
        (self.out/'MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');assert sum(f.stat().st_size for f in self.out.iterdir() if f.is_file())<=FILE_CAP;return r
    def install(self,model):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        allmods=list(model.named_modules());owners=Counter(id(m.quant_method) for _,m in allmods if getattr(m,'quant_method',None) is not None);found=[(kind_of(n),n,m) for n,m in allmods if kind_of(n)]
        assert Counter(k for k,_,_ in found)=={k:v[1] for k,v in GROUPS.items()}
        for kind,name,m in found:
            q=m.quant_method;assert owners[id(q)]==1 and type(q) is (UnquantizedEmbeddingMethod if kind=='head' else UnquantizedLinearMethod)
            assert getattr(m,'bias',None) is None and tuple(m.weight.shape)==GROUPS[kind][2:];assert str(m.weight.dtype)=='torch.bfloat16' and str(m.weight.device)=='cuda:0'
            assert q.apply.__func__ is (UnquantizedEmbeddingMethod.apply if kind=='head' else UnquantizedLinearMethod.apply)
        try:
            for kind,name,m in found:
                q=m.quant_method;original=q.apply;had='apply' in q.__dict__;prior=q.__dict__.get('apply');source=identity(original);fp=(id(m.weight),m.weight.data_ptr(),version(m.weight));self.counts[name]=0
                self.modules.append(dict(kind=kind,path=name,weight=metadata(m.weight),source=source,gemm_source=identity(q._gemm_impl) if getattr(q,'_gemm_impl',None) else None,representative=is_representative(kind,name)))
                def wrapped(this,layer,x,bias=None,_kind=kind,_name=name,_module=m,_original=original,_source=source,_fp=fp):
                    import torch
                    from packed_vllm_runtime_compare_v4.logger import PublicLogger
                    assert layer is _module and bias is None and not self.pending
                    assert (id(layer.weight),layer.weight.data_ptr(),version(layer.weight))==_fp
                    assert x.ndim==2 and x.dtype==torch.bfloat16 and x.stride(1)==1
                    count=self.counts[_name];ph=phase(_kind,count,x.shape[0],x.shape[1]);self.counts[_name]+=1
                    selected=is_representative(_kind,_name) and not any(z['kind']==_kind and z['phase']==ph for z in self.samples)
                    row=dict(kind=_kind,module=_name,call_index=count,phase=ph,X=metadata(x),W=metadata(layer.weight),source=_source,selected=selected,original_calls=0,passed=False)
                    self.rows.append(row);self.pending=True
                    try:
                        if selected:
                            key=_kind+'_'+ph
                            if _kind not in self.weights:self.weights[_kind]=self.weight_link(_kind,layer.weight)
                            sample=dict(kind=_kind,phase=ph,module=_name,source=_source,X=self.tensor(x,key+'_X'),weight_key=_kind,flags_before=[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]);assert sample['flags_before']==[True,True]
                            torch.cuda.synchronize();logger=PublicLogger(self.out/(key+'-cublas.log'))
                            with logger:
                                y=_original(layer,x,bias);row['original_calls']=1;torch.cuda.synchronize()
                            logger.unchanged()
                        else:y=_original(layer,x,bias);row['original_calls']=1
                        assert isinstance(y,torch.Tensor) and y.shape==(x.shape[0],GROUPS[_kind][2]) and y.dtype==torch.bfloat16 and y.stride(1)==1
                        assert y.untyped_storage().data_ptr() not in (x.untyped_storage().data_ptr(),layer.weight.untyped_storage().data_ptr())
                        row['Y']=metadata(y);row['passed']=(id(layer.weight),layer.weight.data_ptr(),version(layer.weight))==_fp;assert row['passed']
                        if selected:
                            sample['X_preserved']=digest(x)==sample['X']['logical_sha256'];assert sample['X_preserved']
                            sample['Y']=self.tensor(y,key+'_Y');sample['public_logger']=logger.record
                            sample['public_api']=parse_one(logger.path,sample['X']['metadata'],metadata(layer.weight),sample['Y']['metadata'])
                            sample['flags_after']=[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert sample['flags_before']==sample['flags_after']==[True,True]
                            sample['returned_native_object']=True;self.samples.append(sample)
                        return y
                    finally:self.pending=False;self.save()
                fn=types.MethodType(wrapped,q);self.saved.append((q,had,prior,fn));q.apply=fn
            return dict(installed=True,modules=self.modules)
        except BaseException:self.close();raise
    def close(self):
        conflicts=[]
        for q,had,prior,fn in reversed(self.saved):
            if q.__dict__.get('apply') is not fn:conflicts.append(type(q).__name__);continue
            if had:q.apply=prior
            else:delattr(q,'apply')
        self.saved=[];self.closed=True;assert not conflicts,conflicts
    def verify(self):self.verify_weights();assert self.save()['complete']
def install(model,out):
    assert not hasattr(model,ATTR);o=Observer(out);setattr(model,ATTR,o)
    try:return o.install(model)
    except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
    o=getattr(model,ATTR,None)
    if o is None:return {'observer_absent':True}
    try:o.close();o.verify();return o.save()
    finally:delattr(model,ATTR)
def close_only(model):
    o=getattr(model,ATTR,None)
    if o is None:return {'observer_absent':True}
    try:o.close();return {'observers_removed':True}
    finally:delattr(model,ATTR)
