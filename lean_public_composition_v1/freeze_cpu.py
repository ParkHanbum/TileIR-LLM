"""Freeze existing public proof pins plus new CPU sources. No device operations."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;OLD=P/'all_gemm_sixteen_sampler_composition_v2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=R/name;assert not p.exists(),str(p);p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');return sha(p)
oldpin=OLD/'PINS.json';pins=json.loads(oldpin.read_text());bad=[p for p,h in pins.items()if not Path(p).is_file()or sha(Path(p))!=h];assert not bad,bad
upstream=[oldpin,OLD/'FINAL-RESULTS.md',OLD/'FINAL-COVERAGE.json',OLD/'FINAL-SEAL.json',OLD/'allGemmSixteenSampler02/RESULT.json',OLD/'allGemmSixteenSampler02-launch.json',OLD/'allGemmSixteenSampler02-recovery.json']
assert sha(upstream[4])=='c45d43017d5596ef12aa2e58ca1d8c29374bbfc650ca0380aaac55bd74051636'
tests=json.loads((R/'CPU-TESTS.json').read_text());assert tests['complete']and tests['actual_exit']==0 and not tests['timed_out']and tests['source_before']==tests['source_after']
assert all(sha(Path(p))==h for p,h in tests['source_after'].items())
local=['__init__.py','support.py','model.py','metadata.py','recurrent.py','sampler.py','adapter.py','test_core_cpu.py','test_sampler_cpu.py','test_recurrent_cpu.py','verify_cpu.py','freeze_cpu.py','PLAN.md','DATAFLOW-AUDIT.md','QUALIFICATION-PLAN.md','BASELINE-SCOPE.md','CPU-TESTS.json','cpu1.stdout','cpu1.stderr']
assert not (R/'cache_v1').exists()and not (R/'launch.py').exists()and not (R/'run.py').exists()
for p in upstream+[R/n for n in local]:pins[str(p)]=sha(p)
ph=save('PINS.json',pins)
receipt=dict(complete=True,phase='CPU_source_preparation_complete_GPU_pending',actual_cpu_test_exit=0,tests=29,pins_count=len(pins),pins_sha256=ph,prior_pins_verified=len(json.loads(oldpin.read_text())),old_proof_result_sha256=sha(upstream[4]),cpu_tests_sha256=sha(R/'CPU-TESTS.json'),source_only=True,launch_script_present=False,cache_created=False,gpu_query=False,gpu_execution=False,cuda_compile=False,selected_boundaries_expected=5648,selected_logical_expected=5680,baseline='current eager vLLM mode0 graphsNONE B32 P53 G16 logprobs5 defaultT/T; not historical optimized baseline',qualification='not GPU qualified; no performance claim',limits=['public API validation and small support guards remain','native attention/GDN prefill/cat remain','no per-call X/Y raw proof or W byte checks in lean execution','no graph/scheduler/precision/runtime setting changes'])
rh=save('CPU-READY.json',receipt)
print(json.dumps(dict(complete=True,pins=len(pins),pins_sha256=ph,ready_sha256=rh,cpu_tests_sha256=receipt['cpu_tests_sha256'])))
