"""Coordinator-only callback; never imports framework until explicitly called."""
from .cases import cases
from .api import Guard

def qualify(torch,on_progress=None):
    from tileir_vllm_post_update_v3.contract import validate_metadata,validate_values
    records=[]
    def snapshot(args):return tuple(x.detach().view(-1).view(torch.uint8).cpu().numpy().tobytes()for x in args)
    for label,d,expected in cases():
        args=(torch.tensor(d['idx'],dtype=torch.int64,device='cuda'),torch.tensor(d['computed'],dtype=torch.int32,device='cuda'),torch.zeros((32,1),dtype=torch.int64,device='cuda'),torch.zeros((32,248320),dtype=torch.int32,device='cuda'),torch.tensor(d['sampled'],dtype=torch.int64,device='cuda').reshape(32,1),torch.tensor(d['ns'],dtype=torch.int32,device='cuda'),torch.tensor(d['nr'],dtype=torch.int32,device='cuda'),torch.tensor(d['q'],dtype=torch.int32,device='cuda'),torch.zeros((32,128),dtype=torch.int32,device='cuda'),torch.tensor(d['total'],dtype=torch.int32,device='cuda'))
        for (s,t),v in d['hist'].items():args[3][s,t]=v
        validate_metadata(args);before=snapshot(args);handle=torch.cuda.current_stream(args[0].device).cuda_stream
        row=dict(case=label,expected=expected,native=[],candidate=[],readonly_checks=0,candidate_stages=[])
        records.append(row)
        if on_progress:on_progress(records)
        for arm in ('native','candidate'):
            for repeat in range(2):
                guard=Guard();error=None
                try:
                    if arm=='native':validate_values(torch,args)
                    else:guard.validate(torch,args)
                except ValueError as e:error=str(e)
                if error!=expected:raise AssertionError((label,arm,error,expected))
                assert snapshot(args)==before,'Input backing changed';row['readonly_checks']+=1
                assert torch.cuda.current_stream(args[0].device).cuda_stream==handle
                row[arm].append(error)
                if arm=='candidate':
                    assert not guard.pending
                    row['candidate_stages'].append(list(guard.stages))
                    assert guard.failed is (expected is not None)
        row['stream_unchanged']=True;row['passed']=True
        if on_progress:on_progress(records)
    torch.cuda.synchronize()
    return dict(complete=True,records=records,scope='Private typed device backing; no original UVA ownership/model/performance claim')
