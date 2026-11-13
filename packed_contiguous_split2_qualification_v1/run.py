"""Five saved real matrices. GPU is opt-in and parent-lock gated; no timing/model."""
import argparse,ctypes,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins,layout
ARMS=('fp32_partial','bf16_partial')
LABELS=['ba_decode','ba_prefill','qkvz_decode','gate_up_prefill','ba_decode_layer1_mismatch']

def validate_manifest(m):
    assert m['schema']=='packed-contiguous-split2-five-actual-v1'
    assert [r['label'] for r in m['cases']]==LABELS
    assert m['arms']==list(ARMS) and m['torch_flags_required']==[True,True]
    assert m['reference']==dict(math_mode=0,algo=99,compute_type=68,pointer_mode=0,alpha=1,beta=0)
    assert m['model_replacement'] is False and m['timing'] is False
    for row in m['cases']:
        assert row['require_default_matches_captured']==(row['label']!='ba_decode_layer1_mismatch')
        for name in ('X','W','capturedY','saved_original'):
            assert Path(row[name]['path']).is_absolute() and len(row[name]['sha256'])==64
    return m['cases']

def passing(c):
    return (c.get('raw_equal') is True and c.get('strict_failures')==0
            and c.get('candidate_nonfinite')==0 and c.get('reference_nonfinite')==0)

def result_gate(r):
    safe=(r.get('finished') is True and r.get('gpu_executed') is True
          and r.get('timing') is False and r.get('adopted') is False
          and r.get('initial_flags')==r.get('final_flags')==[True,True]
          and r.get('source_pins_unchanged') is True and r.get('reference_closed') is True
          and not r.get('cleanup_errors') and not r.get('error'))
    cases=r.get('cases',[]);good=safe and 1<=len(cases)<=5 and [x.get('label') for x in cases]==LABELS[:len(cases)]
    if good:
        good=all(x.get('controls_pass') is True and x.get('inputs_unchanged') is True and x.get('input_backing_unchanged') is True
                 and x.get('math_mode')==0 and x.get('algo')==99
                 and passing(x.get('native_repeat',{}))
                 and len(x.get('original_checks',[]))==2 and all(passing(c) for c in x['original_checks'])
                 and len(x.get('reference_calls',[]))==2 and all(z.get('math_mode')==0 and z.get('pointer_mode')==0 and z.get('algo')==99 and z.get('compute_type')==68 and z.get('alpha')==1 and z.get('beta')==0 for z in x['reference_calls'])
                 and (x.get('label')=='ba_decode_layer1_mismatch' or passing(x.get('captured_comparison',{}))) for x in cases)
    qualified=[]
    for arm in ARMS:
        status=r.get('arms',{}).get(arm,{})
        if status.get('state')=='five_matrix_pass':
            ok=len(cases)==5 and status.get('passed_cases')==LABELS
            for x in cases:
                rows=[z for z in x.get('candidate_records',[]) if z.get('arm')==arm]
                ok=ok and len(rows)==2 and [z.get('repeat') for z in rows]==[0,1] and all(passing(z.get('comparison',{})) and z.get('inputs_unchanged') is True and z.get('input_backing_unchanged') is True and z.get('output_contract_pass') is True for z in rows)
            if ok:qualified.append(arm)
            else:good=False
        elif status.get('state')=='HOLD':
            halt=status.get('halted_at',{});found=[z for x in cases if x.get('label')==halt.get('case') for z in x.get('candidate_records',[]) if z.get('arm')==arm and z.get('repeat')==halt.get('repeat')]
            good=good and len(found)==1 and not passing(found[0].get('comparison',{}))
        else:good=False
    return dict(complete=bool(good),raw_pass=bool(good and qualified),qualified_arms=qualified if good else [],raw_difference_hold=bool(good and not qualified and all(r['arms'][a]['state']=='HOLD' for a in ARMS)))

def compare(reference,candidate,torch):
    assert reference.shape==candidate.shape and reference.dtype==candidate.dtype==torch.bfloat16
    a=reference.contiguous();b=candidate.contiguous();af=a.float();bf=b.float();finite=torch.isfinite(af)&torch.isfinite(bf);error=(af-bf).abs();mask=a.view(torch.int16)!=b.view(torch.int16)
    indices=mask.flatten().nonzero().flatten()[:32].tolist();n=a.shape[1]
    return dict(elements=a.numel(),raw_equal=not bool(mask.any()),raw_mismatches=int(mask.sum()),strict_failures=int(((error>1e-5+1e-5*af.abs())|~finite).sum()),atol=1e-5,rtol=1e-5,reference_nonfinite=int((~torch.isfinite(af)).sum()),candidate_nonfinite=int((~torch.isfinite(bf)).sum()),max_abs_finite=float(error[finite].max()) if bool(finite.any()) else None,first_differences=[dict(index=[i//n,i%n],reference_bits=int(a.flatten()[i].view(torch.int16))&65535,candidate_bits=int(b.flatten()[i].view(torch.int16))&65535) for i in indices])

def direct_default99(torch):
    # Existing public C wrapper owns handle creation/destruction only; its algo-1 call is NOT used.
    from integration.direct_reference_v4 import Reference,BLAS
    ref=Reference(torch)
    try:
        lib=ctypes.CDLL(str(BLAS));V=ctypes.c_void_p;I=ctypes.c_int
        for name,args in [('cublasSetMathMode',[V,I]),('cublasGetMathMode',[V,ctypes.POINTER(I)]),('cublasSetPointerMode_v2',[V,I]),('cublasGetPointerMode_v2',[V,ctypes.POINTER(I)]),('cublasSetStream_v2',[V,V])]:
            f=getattr(lib,name);f.argtypes=args;f.restype=I
        lib.cublasGemmEx.argtypes=[V,I,I,I,I,I,V,V,I,I,V,I,I,V,V,I,I,I,I];lib.cublasGemmEx.restype=I
        ref.check(lib.cublasSetMathMode(ref.h,0));ref.check(lib.cublasSetPointerMode_v2(ref.h,0))
        def call(x,w,y):
            mode=I();pm=I();ref.check(lib.cublasGetMathMode(ref.h,ctypes.byref(mode)));ref.check(lib.cublasGetPointerMode_v2(ref.h,ctypes.byref(pm)));assert (mode.value,pm.value)==(0,0)
            stream=torch.cuda.current_stream(x.device).cuda_stream;ref.check(lib.cublasSetStream_v2(ref.h,stream));a=ctypes.c_float(1);b=ctypes.c_float(0)
            y.fill_(float('nan'));ref.check(lib.cublasGemmEx(ref.h,1,0,w.shape[0],x.shape[0],x.shape[1],ctypes.byref(a),w.data_ptr(),14,w.stride(0),x.data_ptr(),14,x.stride(0),ctypes.byref(b),y.data_ptr(),14,y.stride(0),68,99))
            return dict(math_mode=mode.value,pointer_mode=pm.value,stream=stream,algo=99,compute_type=68,alpha=1,beta=0)
        return ref,call,BLAS
    except BaseException:
        ref.close();raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
    assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');rows=validate_manifest(json.loads(Path(a.manifest).read_text()))
    if not a.run_gpu:
        print(json.dumps(dict(executed=False,cases=LABELS,arms=ARMS,reference='D99 math0; Torch flags read-only T/T',timing=False)));return 0
    fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.fstat(fd);expected=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert (st.st_dev,st.st_ino)==(expected.st_dev,expected.st_ino)
    assert a.output and a.output.is_absolute();a.output.mkdir(exist_ok=False)
    for key,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton')]:
        (a.output/name).mkdir();os.environ[key]=str(a.output/name)
    r=dict(finished=False,complete=False,gpu_executed=False,adopted=False,timing=False,manifest_sha=a.manifest_sha,cases=[],arms={arm:dict(state='active',passed_cases=[]) for arm in ARMS},reference_limitations='New independent HOST handle/workspace, not original runtime handle. Fixed D99 math0; historical F output never the adoption reference.')
    def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
    save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('570s worker bound')));signal.alarm(570)
    ref=None;torch=None;exit_code=1
    try:
        import torch
        from integration.direct_reference_v4 import tensor_sha
        from tileirblas_once_occ2 import linear as original
        from packed_contiguous_split2_v1.api import linear_candidate
        torch.set_num_threads(4)
        flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
        r['initial_flags']=flags();assert r['initial_flags']==[True,True]
        ref,call,libpath=direct_default99(torch);r.update(gpu_executed=True,reference_version=ref.version,reference_library=dict(path=str(libpath),sha256=sha(libpath)));save()
        for row in rows:
            active=[arm for arm in ARMS if r['arms'][arm]['state']=='active']
            if not active:break
            entry=dict(label=row['label'],math_mode=0,algo=99,candidate_records=[],controls_pass=False);r['cases'].append(entry);save();cpu={}
            for key in ('X','W','capturedY','saved_original'):
                v=row[key];assert sha(v['path'])==v['sha256'];t=torch.load(v['path'],map_location='cpu',weights_only=True);assert t.ndim==2 and t.dtype==torch.bfloat16;cpu[key]=t
            assert cpu['X'].shape[1]==cpu['W'].shape[1]
            shape=(len(cpu['X']),len(cpu['W']));assert all(tuple(cpu[k].shape)==shape for k in ('capturedY','saved_original'))
            def restore(t,meta):
                stride,off=layout(meta,t.shape);size=off+(t.shape[0]-1)*stride[0]+t.shape[1];assert size*2<=256*1024*1024
                backing=torch.full((size,),float('nan'),device='cuda',dtype=t.dtype);v=torch.as_strided(backing,t.shape,stride,off);v.copy_(t);return v,backing
            gx,bx=restore(cpu['X'],row['metadata']['X']);gw,bw=restore(cpu['W'],row['metadata']['W']);before=(tensor_sha(gx),tensor_sha(gw));backing_before=(tensor_sha(bx),tensor_sha(bw))
            def preserved():return (tensor_sha(gx),tensor_sha(gw))==before,(tensor_sha(bx),tensor_sha(bw))==backing_before
            gy=torch.empty(shape,device=gx.device,dtype=gx.dtype);references=[];entry['reference_calls']=[]
            for rep in range(2):
                entry['reference_calls'].append(call(gx,gw,gy));torch.cuda.synchronize();references.append(gy.cpu())
            d=references[0];entry['native_repeat']=compare(d,references[1],torch);assert passing(entry['native_repeat'])
            entry['captured_comparison']=compare(cpu['capturedY'],d,torch);entry['captured_is_required_default']=row['require_default_matches_captured']
            if row['require_default_matches_captured']:assert passing(entry['captured_comparison']),'D99 differs from stored default native; stop before candidates'
            entry['original_checks']=[]
            for rep in range(2):
                y=original(gx,gw);torch.cuda.synchronize();old=y.cpu();c=compare(cpu['saved_original'],old,torch);entry['original_checks'].append(c);assert passing(c),'Unchanged TileIR baseline does not reproduce saved output'
            entry['original_vs_default']=compare(d,old,torch);entry['inputs_unchanged'],entry['input_backing_unchanged']=preserved();assert entry['inputs_unchanged'] and entry['input_backing_unchanged'];entry['controls_pass']=True;save()
            for arm in active:
                passed_arm=True
                for rep in range(2):
                    rec=dict(arm=arm,repeat=rep,passed=False);entry['candidate_records'].append(rec);save()
                    y=linear_candidate(gx,gw,mode=arm);torch.cuda.synchronize()
                    rec['output_contract_pass']=(tuple(y.shape)==shape and y.dtype==torch.bfloat16 and y.device==gx.device and y.is_contiguous() and y.storage_offset()==0 and y.untyped_storage().data_ptr() not in (gx.untyped_storage().data_ptr(),gw.untyped_storage().data_ptr()))
                    assert rec['output_contract_pass'];value=y.cpu();rec['output_sha256']=tensor_sha(value);rec['comparison']=compare(d,value,torch);rec['inputs_unchanged'],rec['input_backing_unchanged']=preserved();assert rec['inputs_unchanged'] and rec['input_backing_unchanged']
                    rec['passed']=passing(rec['comparison'])
                    if not rec['passed']:
                        r['arms'][arm].update(state='HOLD',halted_at=dict(case=row['label'],repeat=rep));passed_arm=False
                        failure=a.output/(row['label']+'-'+arm+'-failure.pt');assert value.numel()*value.element_size()<=64*1024*1024;torch.save(value,failure);rec['failure_output']=dict(path=str(failure),sha256=sha(failure));save();break
                    save()
                if passed_arm:r['arms'][arm]['passed_cases'].append(row['label'])
            entry['inputs_unchanged'],entry['input_backing_unchanged']=preserved();assert entry['inputs_unchanged'] and entry['input_backing_unchanged'];save()
            del gx,gw,bx,bw,gy,y,value,references,d,old,cpu
        for arm in ARMS:
            if r['arms'][arm]['state']=='active':
                assert r['arms'][arm]['passed_cases']==LABELS;r['arms'][arm]['state']='five_matrix_pass'
        r['finished']=True
    except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        errors=[]
        def cleanup(stage,fn):
            try:fn()
            except BaseException as e:errors.append(dict(stage=stage,error=repr(e)))
        if torch is not None:
            cleanup('synchronize',lambda:torch.cuda.synchronize())
            def check_flags():
                r['final_flags']=[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert r['final_flags']==r.get('initial_flags')==[True,True]
            cleanup('readonly_flags',check_flags)
        if ref is not None:
            def close():ref.close();r['reference_closed']=True
            cleanup('reference_close',close)
        def pins():verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;r['source_pins_unchanged']=True
        cleanup('source_pins',pins);signal.alarm(0)
        if errors:r['cleanup_errors']=errors
        gate=result_gate(r);r.update(result_gate=gate,complete=gate['complete']);save()
        exit_code=0 if gate['raw_pass'] else (2 if gate['raw_difference_hold'] else 1)
    return exit_code
if __name__=='__main__':raise SystemExit(main())
