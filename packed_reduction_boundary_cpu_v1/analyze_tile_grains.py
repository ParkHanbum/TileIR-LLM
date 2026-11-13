"""Bounded CPU-only arithmetic hypotheses; never imports/calls CUDA APIs."""
import argparse,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent.parent;R=Path(__file__).parent;sys.path.insert(0,str(P))
from packed_first_mismatch_replay_v2.exact import units,round_dot

def round_fp32_units(n):
    # Units2^-266, exact integer nearest-even. Finite limited domain only.
    sign=-1 if n<0 else 1;z=abs(n)
    shift=max(z.bit_length()-24,117);q,rem=divmod(z,1<<shift);half=1<<(shift-1)
    q+=rem>half or (rem==half and q&1)
    result=sign*(q<<shift)
    if abs(result)>((1<<24)-1)<<(127-23+266):raise OverflowError('FP32 overflow outside diagnostic domain')
    return result

def round_bf16_units(n):
    bits=round_dot(n)
    if bits&0x7f80==0x7f80:raise OverflowError('BF16 overflow outside diagnostic domain')
    return units(bits)<<133

def predict(products,parts,reduction):
    assert len(products)%parts==0
    width=len(products)//parts
    chunks=[round_fp32_units(sum(products[i*width:(i+1)*width])) for i in range(parts)]
    z=0
    for partial in chunks:
        if reduction=='serial_output_bf16':z=round_bf16_units(round_fp32_units(z+partial))
        elif reduction=='parallel_bf16':z=round_fp32_units(z+round_bf16_units(partial))
        elif reduction=='parallel_fp32':z=round_fp32_units(z+partial)
        else:raise ValueError('unsupported fixed arithmetic hypothesis')
    return round_dot(z)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    torch.set_num_threads(1)
    Q=P/'packed_contiguous_split2_qualification_v1';manifest=Q/'MANIFEST.json';receipt=Q/'packedSplit2Qual01/RESULT.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='21f7c37df7702d81a1401905329f044bec6312ebc00f86c88e553f87786b7713'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()=='38254fe44e8be5c7a9828d72735c3fed76d6b6cff8a7630335721d105c878213'
    row=json.loads(manifest.read_text())['cases'][0];input_pins={}
    def load(e):
        p=Path(e['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'];input_pins[str(p)]=e['sha256'];return torch.load(p,map_location='cpu',weights_only=True)
    x=load(row['X']);w=load(row['W']);native=load(row['capturedY']);old=load(row['saved_original'])
    assert x.dtype==w.dtype==native.dtype==old.dtype==torch.bfloat16
    assert torch.equal(x,x[:1].expand_as(x)) and torch.equal(native,native[:1].expand_as(native))
    refs={'D_native':native,'old_TileIR':old}
    for arm in json.loads(receipt.read_text())['cases'][0]['candidate_records']:
        refs['GPU_'+arm['arm']]=load(arm['failure_output'])
    refbits={name:[int(t)&65535 for t in value[0].view(torch.int16)] for name,value in refs.items()}
    a=[units(v) for v in x[0].view(torch.int16).tolist()]
    sums=[[aa*units(bb) for aa,bb in zip(a,ww.view(torch.int16).tolist())] for ww in w]
    results=[]
    for parts in (5,10,20,40,80):
        for reduction in ('parallel_bf16','serial_output_bf16','parallel_fp32'):
            bits=[predict(products,parts,reduction) for products in sums]
            results.append(dict(parts=parts,reduction=reduction,predicted_bf16_bits=bits,mismatches_64_channels={name:sum(a!=b for a,b in zip(bits,target)) for name,target in refbits.items()}))
    r=dict(complete=True,cpu_only=True,independent_X_rows=1,channels=64,logical_batch_rows=32,reference='saved D99/math0 native matched in GPU receipt',scope='Exact BF16 partial dot rounded to FP32; not MMA/internal-accumulator emulation. Fixed K grains512/256/128/64/32 (5/10/20/40/80 partitions), three reduction recipes; exploratory, not generality or adoption proof.',gpu_launches=0,gpu_queries=0,compiler_invocations=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_pins=input_pins,reference_bits=refbits,results=results)
    with (R/'CPU-TILE-GRAIN-RESULT.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps([{k:v for k,v in e.items() if k!='predicted_bf16_bits'} for e in results],indent=2))
if __name__=='__main__':main()
