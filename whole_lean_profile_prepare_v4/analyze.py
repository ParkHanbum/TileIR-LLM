"""Pure trace analysis. Imported predecessor algorithms are not modified.
Counts are primary; duration sums are NOT wall time or overlap-free device busy time.
"""
from collections import Counter
from combined_lean_profile_prepare_v1.profile_contract import inspect_events,scope_names
from combined_lean_profile_prepare_v1.attribute import attribute

PHASES=('install','execution','proof','restore','outside','uncorrelated','ambiguous')

def analyze(events,mode,symbols=None,event_roles=None):
    symbols=symbols or {};event_roles=event_roles or {}
    partition=inspect_events(events,mode)
    attribution=attribute(events,mode,symbols)
    names=scope_names(mode)
    ranges={k:next(e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name')==v) for k,v in names.items()}
    def contains(r,e):
        return (r['pid'],r['tid'])==(e.get('pid'),e.get('tid')) and r['ts']<=e['ts'] and e['ts']+e['dur']<=r['ts']+r['dur']
    def hostphase(e):
        p=[k for k,r in ranges.items() if contains(r,e)]
        return p[0] if len(p)==1 else 'ambiguous' if p else 'outside'
    hosts={}
    for e in events:
        if e.get('ph')=='X' and e.get('cat') in ('cuda_runtime','cuda_driver'):
            c=e.get('args',{}).get('correlation')
            if type(c)is int and c>0:hosts.setdefault(c,[]).append(e)
    def gpu_phase(e):
        ps={hostphase(h) for h in hosts.get(e.get('args',{}).get('correlation'),[])}
        return next(iter(ps)) if len(ps)==1 else 'ambiguous' if ps else 'uncorrelated'
    def direction(e):
        # Only explicit profiler direction labels; no shape/ordinal inference.
        name=e.get('name','').lower()
        found=[d for d in ('htod','dtoh','dtod','htoh') if d in name]
        return {'htod':'H2D','dtoh':'D2H','dtod':'D2D','htoh':'H2H'}[found[0]] if len(found)==1 else 'unknown'
    ledger=[];category_inventory=Counter();non_duration=Counter()
    for i,e in enumerate(events):
        cat=e.get('cat','<missing>');category_inventory[cat]+=1
        if e.get('ph')!='X':non_duration[cat]+=1;continue
        if cat=='kernel':kind='compute_kernel';phase=gpu_phase(e)
        elif cat=='gpu_memcpy':kind='memcpy_'+direction(e);phase=gpu_phase(e)
        elif cat=='gpu_memset':kind='memset';phase=gpu_phase(e)
        elif cat in ('cuda_runtime','cuda_driver'):
            kind='cpu_sync' if 'synchroniz' in e.get('name','').lower() else 'runtime_host';phase=hostphase(e)
        elif cat=='user_annotation':kind='annotation';phase=hostphase(e)
        elif cat=='cpu_op':kind='cpu_op';phase=hostphase(e)
        else:kind='unknown_category';phase=hostphase(e)
        proof=event_roles.get(i,{})
        role=proof.get('role','unknown')
        if role not in ('host_transport','model_copy','model_zero_fill','other_model_operation') or not proof.get('source_sha256') or not proof.get('source_site'):
            role='unknown'
        ledger.append(dict(event_index=i,category=cat,name=e.get('name'),phase=phase,kind=kind,duration_us=e.get('dur',0),source_role=role,role_evidence=proof if role!='unknown' else None))
    totals={}
    for x in ledger:
        k=x['phase']+'/'+x['kind'];v=totals.setdefault(k,dict(count=0,duration_us=0));v['count']+=1;v['duration_us']+=x['duration_us']
    counts=Counter();durations=Counter();loaded_count=0;loaded_duration=0.;kernel_evidence=[]
    for row in attribution['rows']:
        ev=symbols.get(row['kernel'],{});label='unknown'
        # Source/symbol match and loaded provenance are separate evidence levels.
        loaded=bool(ev.get('loaded_artifact_sha256') and ev.get('loaded_sha_observed')is True and ev.get('load_evidence'))
        if row['own_tileir_verified'] and ev.get('source_sha256'):
            label='tileir_source_symbol_matched'
            if loaded and ev['loaded_artifact_sha256']==ev.get('artifact_sha256'):
                loaded_count+=1;loaded_duration+=row['duration_us']
        elif ev.get('native_verified')is True and ev.get('public_source_sha256') and ev.get('native_provenance'):
            label='native_verified'
        counts[label]+=1;durations[label]+=row['duration_us']
        kernel_evidence.append(dict(**row,classification=label,loaded_sha_observed=loaded,llvm_observed=ev.get('llvm_observed')is True,embedded_ptx_observed=ev.get('embedded_ptx_observed')is True))
    denominator=partition['request_kernel_count']
    reconciled=sum(counts.values())==denominator
    ready=partition['model_denominator_ready'] and reconciled
    noncompute=[x for x in ledger if x['phase']=='execution' and (x['kind'].startswith('memcpy_')or x['kind']=='memset')]
    return dict(partition=partition,compute_denominator_count=denominator,compute_count=dict(counts),compute_duration_us=dict(durations),loaded_artifact_verified=dict(count=loaded_count,duration_us=loaded_duration),loaded_artifact_all_verified=bool(ready and loaded_count==denominator and denominator>0),kernel_evidence=kernel_evidence,denominator_ready=ready,compute_all_source_symbol_matched_tileir=bool(ready and counts['tileir_source_symbol_matched']==denominator and denominator>0),all_gpu_work_converted_claim=False,all_vllm_apis_converted_claim=False,functional_noncompute_review=noncompute,ledger=ledger,totals=totals,category_inventory=dict(category_inventory),non_duration_inventory=dict(non_duration),limits='Memcpy/memset implementing model copy/zero/fill remain functional residuals. Unknown roles never count as transport exemptions. Host CPU durations overlap nested records; no wall/performance claim. Outside/cross-thread/unresolved kernels block denominator. GPU categories absent from known schema remain visible and require review before any broad claim.')
