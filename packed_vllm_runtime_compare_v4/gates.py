import math,json,hashlib
from pathlib import Path

def public_logprobs(outputs,keys):
    rows=[]
    for key in keys:
        lp=outputs[key].outputs[0].logprobs
        assert lp is not None and len(lp)==16
        rows.append([{str(t):dict(logprob=float(v.logprob),rank=v.rank) for t,v in sorted(step.items())} for step in lp])
    return rows

def compare_public(a,b):
    assert len(a)==len(b)==32
    exact=True;maximum=0.;n=0
    for ra,rb in zip(a,b):
        assert len(ra)==len(rb)==16
        for sa,sb in zip(ra,rb):
            assert set(sa)==set(sb),'Top-logprob token keys differ'
            for k in sa:
                av,bv=sa[k]['logprob'],sb[k]['logprob']
                assert math.isfinite(av) and math.isfinite(bv)
                assert sa[k]['rank']==sb[k]['rank'],'Token ranks differ'
                error=abs(av-bv);maximum=max(maximum,error);n+=1;exact &= av==bv
                assert error<=1e-5+1e-5*abs(av),'Public logprob mismatch'
    return dict(passed=True,exact_numeric=exact,max_abs=maximum,values=n,tolerance='1e-5+1e-5*abs(baseline)',scope='Public top5 plus chosen-token logprobs; not full-vocabulary logits or BF16 raw intermediates')

def qualification(root):
    # Explicit diagnostic evidence definition; the original default-mode raw HOLD remains unchanged.
    r=json.loads((Path(root)/'QUALIFICATION.json').read_text())
    assert r['definition']=='FP32-functional-compatibility-diagnostic-v1'
    docs={}
    for key,item in r['evidence'].items():
        p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        docs[key]=json.loads(p.read_text())
    base,precision=docs['qualification_result'],docs['precision_result']
    manifest='719a7fb82ab3f9be0a9ee8383226a47811bf5ba8686542347a0c3b6e35bd6a08'
    assert base['manifest_sha']==precision['manifest_sha']==manifest
    assert base['complete'] is True and base['raw_all'] is False and base['numerical_all'] is True
    assert base['capture_contract_pass'] is True and len(base['cases'])==8
    for c in base['cases']:
        assert c['contract_pass'] is True and c['numerical_all'] is True
        assert c['packed_reference_raw'] is (not (c['kind']=='ba' and c['phase']=='decode'))
    assert precision['complete'] is True and precision['gpu_executed'] is True
    assert docs['qualification_launch']['returncode']==2 and docs['precision_launch']['returncode']==0
    assert {c['label'] for c in precision['cases']}=={'ba','K','V'}
    ba=next(c for c in precision['cases'] if c['label']=='ba')
    assert ba['shape']==[32,64,2560] and ba['math_modes']=={'default':0,'no_reduced_reduction':16}
    assert ba['inputs_unchanged'] is True
    assert ba['comparisons']['tileir_vs_default']['raw_bit_mismatches']==768
    assert ba['comparisons']['tileir_vs_no_reduced_reduction']['raw_bit_equal'] is True
    assert ba['oracle_metrics']['tileir']['rounded']['raw_bit_equal'] is True
    assert ba['oracle_metrics']['no_reduced_reduction']['rounded']['raw_bit_equal'] is True
    for key in ('qualification','precision'):
        launch=docs[key+'_launch'];recovery=docs[key+'_recovery']
        assert launch['child_absent'] is True and not launch['remaining_process_group']
        assert launch['result_sha256']==r['evidence'][key+'_result']['sha256']
        assert recovery['lock_available'] is True and recovery['child_pid_absent'] is True
    return r

def report_public_difference(a,b):
    # Informational D/F comparison: preserve raw numerical values and do not silently gate these equal.
    assert len(a)==len(b)==32
    key_differences=rank_differences=numeric_differences=0;maximum=0.;common=0
    for ra,rb in zip(a,b):
        assert len(ra)==len(rb)==16
        for sa,sb in zip(ra,rb):
            key_differences+=set(sa)!=set(sb)
            for k in set(sa)&set(sb):
                av,bv=sa[k]['logprob'],sb[k]['logprob'];assert math.isfinite(av) and math.isfinite(bv)
                common+=1;numeric_differences+=av!=bv;rank_differences+=sa[k]['rank']!=sb[k]['rank'];maximum=max(maximum,abs(av-bv))
    return dict(common_values=common,key_set_differences=key_differences,rank_differences=rank_differences,numeric_differences=numeric_differences,max_abs=maximum,scope='Informational public top5/chosen-token float values; not full logits or raw BF16 equality')
