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

def guard_imports(tracked):
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).is_absolute() and Path(filename).resolve().is_relative_to(REPO):check_file(filename,tracked)
    finder=TrackedImports(tracked);sys.meta_path.insert(0,finder)
    def audit(event,args):
        if event=='exec':
            filename=args[0].co_filename
            if Path(filename).is_absolute() and Path(filename).resolve().is_relative_to(REPO):check_file(filename,tracked)
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
