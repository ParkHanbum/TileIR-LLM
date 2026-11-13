"""Git/source admission and isolated artifact contract (stdlib only)."""
import hashlib, importlib.abc, importlib.machinery, json, os, subprocess, sys
from pathlib import Path
REPO=Path(__file__).resolve().parent.parent
BASE='0b9611661ef216c35fb9f4c5da92662e5930dcbb'
TAG='v7-own-kernel-fixed-workload-20261113'
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args]).decode().strip()
def external_root(value, fresh=False):
    p=Path(value)
    assert p.is_absolute() and p==p.resolve(), 'Artifact root must be absolute and symlink-free'
    assert not p.is_relative_to(REPO) and not REPO.is_relative_to(p), 'Artifact root overlaps checkout'
    if fresh:assert not p.exists(), 'Artifact root already exists'
    return p

def tracked_map():
    result={}
    for row in subprocess.check_output(['git','-C',str(REPO),'ls-tree','-rz','HEAD']).split(b'\0'):
        if not row:continue
        head,name=row.split(b'\t',1);mode,kind,oid=head.decode().split()
        assert mode in ('100644','100755') and kind=='blob','No symlink/submodule sources'
        result[name.decode()]=oid
    return result

def check_file(path,tracked):
    p=Path(path).resolve();rel=str(p.relative_to(REPO))
    assert rel in tracked, 'Ignored/untracked local code: '+rel
    data=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert actual==tracked[rel], 'HEAD bytes differ: '+rel

def verify(expected, external=False):
    assert len(expected)==40 and git('rev-parse','HEAD')==expected,'Wrong HEAD'
    assert git('rev-parse',TAG+'^{commit}')==BASE,'Baseline tag moved'
    subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor',BASE,expected],check=True)
    assert not git('diff','--name-only','HEAD','--'),'Dirty tracked worktree/index'
    tracked=tracked_map()
    for rel in tracked:check_file(REPO/rel,tracked)
    baseline=json.loads((REPO/'baseline_v7/SOURCE-ALLOWLIST.json').read_text())
    for rel,digest in baseline.items():assert sha(REPO/rel)==digest,('Original baseline changed',rel)
    if external:
        for rel,entry in json.loads((REPO/'baseline_v7/BASELINE.json').read_text())['proofs'].items():
            assert sha(REPO/rel)==entry['sha256'],('Baseline proof changed',rel)
        for path,entry in json.loads((REPO/'baseline_v7/EXTERNAL-EVIDENCE.json').read_text()).items():
            assert sha(path)==entry['sha256'],('External evidence drift',path)
    return tracked

class TrackedImports(importlib.abc.MetaPathFinder):
    def __init__(self,tracked):self.tracked=tracked
    def find_spec(self,fullname,path=None,target=None):
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.origin and Path(spec.origin).is_absolute():
            p=Path(spec.origin).resolve()
            if p.is_relative_to(REPO):check_file(p,self.tracked)
        return None

POST_GUARD_SOURCE=REPO/'residual_four_gdn_owned_lean_v2/four.py'
POST_GUARD_GENERATOR=REPO/'public_post_update_guard_v1/binding.py'
POST_GUARD_VIRTUAL=str(POST_GUARD_SOURCE)+'.post_update_guard_copy'

def reviewed_post_guard_exec(code,caller,tracked):
    """One exact baseline factory, source and AST transform; not a suffix rule."""
    import ast,textwrap,types
    assert code.co_filename==POST_GUARD_VIRTUAL,'Unknown virtual filename'
    check_file(POST_GUARD_SOURCE,tracked);check_file(POST_GUARD_GENERATOR,tracked)
    binding=sys.modules.get('public_post_update_guard_v1.binding')
    four=sys.modules.get('residual_four_gdn_owned_lean_v2.four')
    assert binding is not None and four is not None,'Missing original factory modules'
    assert caller.f_globals is vars(binding) and caller.f_code is binding.make_method.__code__,'Foreign virtual-code generator'
    assert caller.f_code.co_filename==str(POST_GUARD_GENERATOR) and caller.f_code.co_name=='make_method'
    generator=compile(POST_GUARD_GENERATOR.read_text(),str(POST_GUARD_GENERATOR),'exec',dont_inherit=True)
    expected_generator=next(x for x in generator.co_consts if isinstance(x,types.CodeType) and x.co_name=='make_method')
    assert caller.f_code==expected_generator,'Generator code differs from tracked source'
    assert binding.SOURCE==POST_GUARD_SOURCE,'Generator source changed'
    original=caller.f_locals['original'];fn=getattr(original,'__func__',original)
    assert fn is four.Replacement.call and fn.__code__.co_filename==str(POST_GUARD_SOURCE),'Foreign original method'
    assert fn.__globals__ is vars(four),'Foreign original namespace'
    text=POST_GUARD_SOURCE.read_text();tree=ast.parse(text)
    original_module=compile(text,str(POST_GUARD_SOURCE),'exec',dont_inherit=True)
    original_class=next(x for x in original_module.co_consts if isinstance(x,types.CodeType) and x.co_name=='Replacement')
    original_call=next(x for x in original_class.co_consts if isinstance(x,types.CodeType) and x.co_name=='call')
    assert fn.__code__==original_call,'Original method code differs from tracked source'
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Replacement')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='call')
    # Preserve the same dedented inspect-source line positions used by make_method.
    node=ast.parse(textwrap.dedent(''.join(text.splitlines(keepends=True)[method.lineno-1:method.end_lineno]))).body[0]
    matches=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validate_values']
    assert len(matches)==1 and not matches[0].keywords
    assert [ast.dump(x,include_attributes=False) for x in matches[0].args]==[ast.dump(ast.Name(id=n,ctx=ast.Load()),include_attributes=False) for n in ('torch','args')]
    matches[0].func.id='_post_update_value_guard'
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    assert ast.dump(caller.f_locals['module'],include_attributes=True)==ast.dump(module,include_attributes=True),'Unapproved AST transformation'
    expected=compile(module,POST_GUARD_VIRTUAL,'exec',dont_inherit=True)
    assert code==expected and code.co_linetable==expected.co_linetable and code.co_exceptiontable==expected.co_exceptiontable,'Transformed bytecode mismatch'

def check_execution(code,caller,tracked):
    filename=code.co_filename
    if filename==POST_GUARD_VIRTUAL:
        reviewed_post_guard_exec(code,caller,tracked)
    elif Path(filename).is_absolute() and Path(filename).resolve().is_relative_to(REPO):
        check_file(filename,tracked)

def guard_imports(tracked):
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).is_absolute() and Path(filename).resolve().is_relative_to(REPO):check_file(filename,tracked)
    finder=TrackedImports(tracked);sys.meta_path.insert(0,finder)
    def audit(event,args):
        if event=='exec':
            check_execution(args[0],sys._getframe(1),tracked)
    sys.addaudithook(audit)
    return finder

def qualification(root):
    root=external_root(root)
    expected=os.environ['V7_REGRESSION_HEAD']
    verify(expected,external=True)
    assert json.loads((root/'REGRESSION.json').read_text())['head']==expected
    from whole_lean_profile_prepare_v4.admission import require_ready
    return dict(require_ready(),regression_head=expected,original_tag=TAG,artifact_root=str(root))

def prepare(root,expected):
    tracked=verify(expected,external=True);root=external_root(root,fresh=True)
    root.mkdir(parents=True,exist_ok=False)
    pins={str(REPO/rel):sha(REPO/rel) for rel in tracked}
    (root/'PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
    (root/'REGRESSION.json').write_text(json.dumps(dict(head=expected,baseline=BASE,performance=False),indent=2)+'\n')
    return root

CACHE_MANIFEST=REPO/'whole_v7_public_blas_logger_v4_process0/CACHE-PRESEED-01.json'
CACHE_MANIFEST_SHA='5f0a0d866e2a23f070889d7f5bd33eafe704a3510914d6915bad3e41a7e54ddf'
def prepare_cache(root):
    """Explicit copy only. Never deletes/reuses a destination or starts GPU."""
    import shutil
    root=external_root(root);qualification(root)
    assert sha(CACHE_MANIFEST)==CACHE_MANIFEST_SHA,'Original cache manifest changed'
    old=json.loads(CACHE_MANIFEST.read_text());source=Path(old['source']);dest=root/'cache_v1'
    assert not dest.exists() and not (root/'CACHE-PRESEED-01.json').exists()
    assert old['count']==1033 and old['bytes']==57498718
    files={}
    for name,digest in old['source_files'].items():
        p=Path(name);rel=p.relative_to(source) if p.is_absolute() else p
        assert not rel.is_absolute() and '..' not in rel.parts
        src=source/rel;assert src.is_file() and not src.is_symlink() and src.resolve().is_relative_to(source.resolve())
        assert sha(src)==digest;files[str(rel)]=digest
    assert len(files)==1033 and {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}==set(files)
    dest.mkdir(exist_ok=False)
    for rel,digest in files.items():
        src=source/rel;dst=dest/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as inp,dst.open('xb') as out:shutil.copyfileobj(inp,out)
        assert sha(src)==sha(dst)==digest
        assert (src.stat().st_dev,src.stat().st_ino)!=(dst.stat().st_dev,dst.stat().st_ino)
    receipt=dict(source=str(source),destination=str(dest),source_files=files,count=len(files),bytes=sum((dest/p).stat().st_size for p in files),same_bytes=True,distinct_inodes=True,source_unchanged=True,source_receipt=str(CACHE_MANIFEST),source_receipt_sha256=CACHE_MANIFEST_SHA)
    assert receipt['bytes']==old['bytes']
    with (root/'CACHE-PRESEED-01.json').open('x') as f:json.dump(receipt,f,indent=2)
    from all_gemm_sixteen_sampler_composition_v2.cache_gate import preflight
    return preflight(root)
