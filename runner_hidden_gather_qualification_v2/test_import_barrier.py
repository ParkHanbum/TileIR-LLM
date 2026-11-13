import unittest,subprocess,sys,tempfile,json,os
from pathlib import Path
R=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_actual_non_dry_before_torch(self):
  manifest=R.parent/'runner_hidden_gather_runtime_capture_v1/hiddenGatherCapture01/bank/MANIFEST.json'
  import hashlib
  script="""import builtins,sys,runpy,os,json
from pathlib import Path
real=builtins.__import__
def blocked(name,*a,**k):
 if name=='torch':
  out=Path(sys.argv[sys.argv.index('--output')+1]);pairs={'CUDA_TILE_CACHE_DIR':'tile','CUDA_TILE_TEMP_DIR':'tiletmp','CUDA_CACHE_PATH':'cuda','TMPDIR':'tmp'}
  assert all(os.environ[x]==str((out/n).resolve()) and (out/n).is_dir() for x,n in pairs.items())
  (out/'IMPORT-BARRIER.json').write_text(json.dumps(dict(passed=True,env={k:os.environ[k] for k in pairs})))
  raise RuntimeError('EXPECTED_CPU_IMPORT_BARRIER')
 if name.startswith('cuda'):raise AssertionError('CUDA import before torch barrier')
 return real(name,*a,**k)
builtins.__import__=blocked
runpy.run_path(sys.argv.pop(1),run_name='__main__')
"""
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run'
   with (Path(d)/'fd').open('w') as f:
    env=dict(os.environ,ALL_GEMM_LOCK_FD=str(f.fileno()),CUDA_TILE_CACHE_DIR='/not-used-old-cache',CUDA_TILE_TEMP_DIR='/not-used-old-temp',CUDA_CACHE_PATH='/not-used-old-driver',TMPDIR='/not-used-old-tmp')
    p=subprocess.run([sys.executable,'-B','-c',script,str(R/'run.py'),'--run-gpu','--manifest',str(manifest),'--manifest-sha',hashlib.sha256(manifest.read_bytes()).hexdigest(),'--output',str(out)],env=env,cwd=R.parent,pass_fds=(f.fileno(),),capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertIn('EXPECTED_CPU_IMPORT_BARRIER',p.stderr);self.assertTrue(json.loads((out/'IMPORT-BARRIER.json').read_text())['passed'])
   result=json.loads((out/'RESULT.json').read_text());self.assertFalse(result['complete']);self.assertEqual(len(result['cache_environment']),4)
if __name__=='__main__':unittest.main()
