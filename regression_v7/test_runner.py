"""Intentional gate failures and source-only replay equivalence."""
import ast, hashlib, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from . import common, whole

class Gates(unittest.TestCase):
    def test_wrong_head(self):
        with patch.object(common,'git',return_value='b'*40):
            with self.assertRaises(AssertionError):common.verify('a'*40)
    def test_changed_tag(self):
        with patch.object(common,'git',side_effect=['a'*40,'b'*40]):
            with self.assertRaises(AssertionError):common.verify('a'*40)
    def test_inside_artifact(self):
        with self.assertRaises(AssertionError):common.external_root(common.REPO/'new-artifacts')
    def test_collision(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):common.external_root(d,fresh=True)
    def test_ignored_import(self):
        with self.assertRaises(AssertionError):common.check_file(__file__,{})
    def test_tracked_changed(self):
        with self.assertRaises(AssertionError):common.check_file(__file__,{str(Path(__file__).resolve().relative_to(common.REPO)):'0'*40})
    def test_request_ast(self):
        original=(whole.LEGACY/'run.py').read_text()
        self.assertEqual(whole.request_ast(original),whole.request_ast(whole.worker_source()))
    def test_launcher_scope(self):
        source=whole.launcher_source();ast.parse(source)
        for text in ('timeout=930','stop_child(child,rec)',"rec['after']=h.admit()","audit(result,root=R)",'fcntl.LOCK_EX|fcntl.LOCK_NB'):
            self.assertIn(text,source)
        self.assertIn('regression_v7.cli',source)
        self.assertNotIn('from tileir_cat_prepared_qual_v1',source)
    def test_unknown_transform(self):
        with self.assertRaises(AssertionError):whole.replace_once('x x','x','y')
    def test_no_elf_name_heuristic(self):
        with self.assertRaises(AssertionError):whole.elf_symbols(b'cutile fake')
    def test_no_performance(self):
        s=whole.worker_source()
        self.assertIn('whole_lean_extensions_runtime_v7 import adapter',s)
        self.assertNotIn('prequalified_v7_',s)
        self.assertNotIn('body_hoist',s)
    def test_module_receipt_tampering(self):
        import copy,os
        from . import cli
        from .test_gemm import result_fixture
        r=dict(complete=True,gpu_executed=True,precision_restored=True,suite='gemm',modules={'gemm':result_fixture()},dependencies={},head='a'*40,cache_preseed={'destination':'/tmp/regression-test/cache_v1'},final_sync_attempted=True,final_sync_succeeded=True,precision_policy=dict(initial=dict(bf16=True,split_k=True),restored=True,restored_split_k=True))
        with patch.dict(os.environ,{'V7_REGRESSION_HEAD':'a'*40}),patch.object(cli,'dependencies',return_value={}):
            self.assertTrue(cli.module_audit(r,Path('/tmp/regression-test'))['passed'])
            for kind in ('missing','duplicate','raw','precision','suite','census','sync'):
                bad=copy.deepcopy(r)
                if kind=='missing':bad['modules']['gemm']['cases'].pop()
                elif kind=='duplicate':bad['modules']['gemm']['cases'][1]=bad['modules']['gemm']['cases'][0]
                elif kind=='raw':bad['modules']['gemm']['cases'][0]['candidate_comparisons'][0]['raw_equal']=False
                elif kind=='precision':bad['precision_policy']['initial']['bf16']=False
                elif kind=='suite':bad['suite']='modules'
                elif kind=='census':bad['modules']['gemm']['gpu_candidate_calls']=0
                else:bad['final_sync_succeeded']=False
                self.assertFalse(cli.module_audit(bad,Path('/tmp/regression-test'))['passed'],kind)
    def test_actual_dynamic_gate_construction(self):
        # Uses production AST copier over actual public gate sources; stdlib only.
        from whole_lean_extensions_runtime_v7 import integer_gate
        dispatch,audit=integer_gate.build()
        self.assertTrue(callable(dispatch) and callable(audit))
    def test_failure_sync_attempted(self):
        import json,os,sys
        from types import SimpleNamespace as N
        from . import cli
        from all_gemm_sixteen_sampler_composition_v2 import cache_gate
        calls=[]
        fake=N(cuda=N(synchronize=lambda:calls.append('sync')),backends=N(cuda=N(matmul=N(allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True))))
        def fail(*args):raise ValueError('intentional module failure')
        target=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat()
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'torch':fake}),patch.dict(os.environ,{'GEMM_COMPOSED_LOCK_FD':'999','VLLM_COMPARISON_PARENT_APPROVED':'1','V7_REGRESSION_HEAD':'a'*40}),patch.object(cli.common,'qualification'),patch.object(cli,'dependencies',return_value={}),patch.object(cli.os,'fstat',return_value=target),patch.object(cache_gate,'preflight',return_value={}),patch.object(cli.importlib,'import_module',return_value=N(run_gpu=fail)),patch.object(cli.signal,'signal'),patch.object(cli.signal,'alarm'):
            with self.assertRaises(ValueError):cli.module_worker(Path(d),Path(d)/'out','gemm')
            result=json.loads((Path(d)/'out/RESULT.json').read_text())
            self.assertFalse(result['complete']);self.assertTrue(result['final_sync_succeeded']);self.assertEqual(calls,['sync'])
    def test_original_launcher_explicit_root_port(self):
        source=common.REPO/'whole_lean_extensions_runtime_v7/launch.py'
        tree=ast.parse(source.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='audit']
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].keywords[0].arg,'root')
        self.assertEqual(ast.dump(calls[0].keywords[0].value),"Name(id='R', ctx=Load())")
    def test_only_exact_historical_test_skipped(self):
        from . import cli
        suite=cli.cpu_suite(cli.HISTORICAL_SOURCE_ONLY_TEST)
        result=unittest.TestResult();suite.run(result)
        self.assertEqual(result.testsRun,1);self.assertEqual(len(result.skipped),1)
        self.assertEqual(result.skipped[0][0].id(),cli.HISTORICAL_SOURCE_ONLY_TEST)
        self.assertIn('preserved historical',result.skipped[0][1])
        with patch.object(cli.unittest.defaultTestLoader,'loadTestsFromName',return_value=unittest.TestSuite()):
            with self.assertRaises(AssertionError):cli.cpu_suite('whole_lean_extensions_runtime_v7.test_cpu')
    def test_nonwhole_result_rejected(self):
        from .cli import module_audit
        self.assertFalse(module_audit({'complete':True})['passed'])
if __name__=='__main__':unittest.main()
