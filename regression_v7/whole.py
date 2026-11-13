"""Owned adaptation of the proven launcher/request. Never edits legacy files."""
import ast, json, os, signal, sys, time
from pathlib import Path
from .common import REPO, external_root, qualification, sha
LEGACY=REPO/'whole_v7_public_blas_logger_v4'

def replace_once(source, old, new):
    assert source.count(old)==1, ('Unexpected original source',old)
    return source.replace(old,new,1)

def request_ast(source):
    tree=ast.parse(source)
    return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='request'),include_attributes=False)

def worker_source():
    original=(LEGACY/'run.py').read_text();s=original
    s=replace_once(s,"assert R.parent==P and R.name in tuple(SOURCE_ROOT.name+'_process'+str(i) for i in range(3))",'assert R == external_root(os.environ["V7_REGRESSION_ROOT"])')
    s=replace_once(s,'from whole_v7_public_blas_logger_v4.gates import qualification','from regression_v7.common import qualification, external_root')
    # Root validation precedes this import in the original, so inject only the
    # pure path helper at module scope; all numerical statements are untouched.
    s=replace_once(s,'from functools import partial','from functools import partial\nfrom regression_v7.common import external_root')
    assert request_ast(s)==request_ast(original)
    return s

def group_members(pgid):
    members=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid:members.append(dict(pid=int(entry.name),state=fields[0],start_ticks=fields[19]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return members

def stop_child(child,record):
    if child is None:return
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if child.poll() is not None and not group_members(child.pid):break
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:pass
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            child.poll()
            if child.returncode is not None and not group_members(child.pid):break
            time.sleep(.1)
    record.update(returncode=child.poll(),remaining_process_group=group_members(child.pid),child_absent=not Path(f'/proc/{child.pid}').exists())

def launcher_source():
    s=(LEGACY/'launch.py').read_text()
    s=replace_once(s,'from tileir_cat_prepared_qual_v1.launch_raw_all_v1 import stop_child','from regression_v7.whole import stop_child')
    s=replace_once(s,'from whole_v7_public_blas_logger_v4.gates import qualification','from regression_v7.common import qualification, external_root')
    s=replace_once(s,"R=SOURCE_ROOT.parent/(SOURCE_ROOT.name+'_process'+str(args.process_index))",'R=external_root(os.environ["V7_REGRESSION_ROOT"])')
    s=replace_once(s,"cmd=[PYTHON,'-B',str(SOURCE_ROOT/'run.py'),'--run-gpu','--output',str(run)]", "cmd=[PYTHON,'-B','-m','regression_v7.cli','--child','--suite',os.environ['V7_REGRESSION_SUITE'],'--head',os.environ['V7_REGRESSION_HEAD'],'--artifact-root',str(R),'--output',str(run)]")
    return s

def execute_owned(source,filename,args):
    old=sys.argv;sys.argv=[str(filename),*args]
    try:
        scope={'__name__':'__main__','__file__':str(filename),'__package__':None}
        exec(compile(source,str(filename),'exec'),scope)
    finally:sys.argv=old

def launch(root,suite):
    os.environ.update(V7_REGRESSION_ROOT=str(external_root(root)),V7_REGRESSION_SUITE=suite)
    source=launcher_source()
    if suite!='whole':
        source=replace_once(source,'from whole_v7_public_blas_logger_v4.gates import audit','from regression_v7.cli import module_audit as audit')
    execute_owned(source,LEGACY/'launch.py',['--run-gpu','--process-index','0','--run-name','regression01'])

def worker(root,output):
    assert str(root)==os.environ['PUBLIC_BLAS_AUDIT_ROOT']
    execute_owned(worker_source(),LEGACY/'run.py',['--run-gpu','--output',str(output)])

def elf_symbols(b):
    import struct
    assert b[:6]==b'\x7fELF\x02\x01'
    shoff=struct.unpack_from('<Q',b,40)[0];size,num=struct.unpack_from('<HH',b,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',b,shoff+i*size) for i in range(num)]
    names=set()
    for s in sections:
        if s[1] not in (2,11):continue
        st=sections[s[6]];strings=b[st[4]:st[4]+st[5]]
        for pos in range(s[4],s[4]+s[5],s[9]):
            name,info,other,ndx,value,sz=struct.unpack_from('<IBBHQQ',b,pos)
            if info&15==2 and ndx and name:names.add(strings[name:strings.find(b'\0',name)].decode())
    return names

def analyze(root):
    """Own cache only; full ELF symbol+bytecode+HEAD source, not loaded SHA."""
    import hashlib,re,sqlite3
    from whole_lean_profile_prepare_v4.analyze import analyze as analyze_events
    from whole_v7_public_blas_logger_v4.gates import audit
    root=external_root(root);run=root/'regression01';out=root/'analysis';out.mkdir(exist_ok=False)
    result=json.loads((run/'RESULT.json').read_text());assert audit(result,root)['passed']
    receipt=json.loads((root/'regression01-launch.json').read_text())
    assert receipt['returncode']==0 and receipt['qualification_complete'] and receipt['child_absent'] and receipt['lock_reacquired'] and receipt.get('after') and not receipt.get('remaining_process_group') and 'error' not in receipt
    recovery=json.loads((root/'RECOVERY.json').read_text())
    assert recovery['complete'] and recovery['head']==os.environ['V7_REGRESSION_HEAD']
    assert recovery['actual_outer_exit']==0 and recovery['actual_child_exit']==0 and recovery['child_absent'] and not recovery['remaining_process_group'] and recovery['lock_available'] and recovery['post']
    assert (root/'OUTER-EXIT.txt').read_text()=='EXIT=0\n'
    assert recovery['launch_sha256']==sha(root/'regression01-launch.json') and recovery['result_sha256']==sha(run/'RESULT.json') and recovery['outer_exit_sha256']==sha(root/'OUTER-EXIT.txt')
    pins=json.loads((root/'PINS.json').read_text());records=[]
    digest=lambda b:hashlib.sha256(b).hexdigest()
    with sqlite3.connect('file:'+str(root/'cache_v1/tile/cache.db')+'?mode=ro',uri=True) as conn:
        for key,blob,meta in conn.execute('select key,blob,metadata from cache'):
            m=json.loads(meta);names=m['kernel_names'];assert set(names)<=elf_symbols(blob)
            records.append(dict(key=key,cubin_sha256=digest(blob),full_symbols=names))
    byhash={r['cubin_sha256']:r for r in records};verified={}
    for cubin in (root/'cache_v1/tiletmp').glob('*.cubin'):
        r=byhash.get(sha(cubin));bc=cubin.with_suffix('.bytecode')
        if r is None or not bc.is_file():continue
        data=bc.read_bytes();paths=[]
        for hit in re.finditer(re.escape(str(REPO).encode()+b'/'),data):
            match=re.match(rb'[A-Za-z0-9_./-]+?\.py',data[hit.start():])
            if match:paths.append(match.group().decode())
        sources=[]
        for path in sorted(set(paths)):
            assert pins.get(path)==sha(path),('Unpinned generated source',path)
            sources.append(dict(path=path,sha256=sha(path)))
        assert all(sym.encode() in data for sym in r['full_symbols'])
        r.update(own_cubin_path=str(cubin),bytecode_sha256=sha(bc),pinned_sources=sources)
        for symbol in r['full_symbols']:
            entry=symbol.split('_Kt')[0]
            candidates=[s for s in sources if re.search(r'^def '+re.escape(entry)+r'\(',Path(s['path']).read_text(),re.M)]
            if not sources:continue
            assert len(candidates)==1,(symbol,candidates)
            s=candidates[0]
            verified[symbol]=dict(verified_own_elf=True,artifact_sha256=r['cubin_sha256'],source_file=s['path'],source_sha256=s['sha256'],entry=entry,bytecode_sha256=sha(bc),cache_key=r['key'],loaded_sha_observed=False)
    (out/'OWN-MAP.json').write_text(json.dumps(dict(records=records,verified_full_symbols=verified),indent=2)+'\n')
    summaries={}
    for mode in ('native','lean'):
        r=analyze_events(json.loads((run/(mode+'.trace.json')).read_text())['traceEvents'],mode,verified)
        (out/(mode+'.json')).write_text(json.dumps(r,indent=2)+'\n');summaries[mode]={k:r[k] for k in ('compute_denominator_count','compute_count','totals','loaded_artifact_verified')}
        if mode=='lean':
            assert r['denominator_ready'] and r['compute_denominator_count']==13300
            assert r['compute_all_source_symbol_matched_tileir']
            assert r['totals'].get('execution/memcpy_D2D',{}).get('count',0)==0
            assert r['totals'].get('execution/memset',{}).get('count',0)==0
    link_path=REPO/'three_process_own_call_audit_link_v1/LINK.json'
    assert sha(link_path)=='00a791040f72ab420d3b657dad5873467e0fd0c3f12d06f20f4f2e6a84ce7ba9'
    link=json.loads(link_path.read_text());oldsets=[{r['artifact_sha256'] for r in p['records']} for p in link['processes']]
    assert len(oldsets)==3 and all(s==oldsets[0] and len(s)==172 for s in oldsets)
    assert link['static_call_sites_per_set']==65 and link['static_call_sites_have_local_targets'] is True
    current={r['cubin_sha256'] for r in records if r.get('pinned_sources')}
    call_proof=dict(exact_prior_set=current==oldsets[0],new_cubins=sorted(current-oldsets[0]),missing_cubins=sorted(oldsets[0]-current),prior_link_sha256=sha(link_path),static_internal_call_sites=65 if current==oldsets[0] else None,loaded_module_sha_observed=False)
    final=dict(passed=call_proof['exact_prior_set'],internal_call_audit=call_proof,head=os.environ['V7_REGRESSION_HEAD'],result_sha256=sha(run/'RESULT.json'),same_run_endpoint_and_logger=True,summaries=summaries,loaded_sha_observed=False,performance_run=False,all_gpu_work_claim=False)
    (out/'FINAL.json').write_text(json.dumps(final,indent=2)+'\n')
    assert final['passed'],'New own artifact set: internal-call proof pending, no automatic disassembly or PASS'
    return final
