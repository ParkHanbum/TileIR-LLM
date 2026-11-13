"""Actual inspect.getsource path with filename/line-preserving compiled public method.
The function body is NEVER invoked. Only stdlib and a fake annotation type are used.
"""
import ast
import copy
import inspect
from pathlib import Path
import textwrap
import types
import unittest
from unittest.mock import patch
from .runtime_adapter import source_checked
PATH=Path('/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py')
MODULE='vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn'

def fixture():
    tree=ast.parse(PATH.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenGatedDeltaNetAttention')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_cuda')
    cls=copy.deepcopy(cls); cls.bases=[];cls.keywords=[];cls.decorator_list=[];cls.body=[copy.deepcopy(fn)]
    Tensor=type('Tensor',(),{'__module__':'torch'})
    namespace={'__name__':MODULE,'torch':types.SimpleNamespace(Tensor=Tensor)}
    # Retain ORIGINAL class/method line offsets and exact real filename for inspect.
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(PATH),'exec'),namespace)
    owner=namespace['QwenGatedDeltaNetAttention']()
    owner._forward_method=owner.forward_cuda
    return fn,owner

class Tests(unittest.TestCase):
    def test_full_unbound_and_cached_bound_path(self):
        original,owner=fixture()
        self.assertEqual(owner.forward_cuda.__func__.__code__.co_firstlineno,original.lineno)
        self.assertEqual(source_checked(owner.forward_cuda.__func__),PATH.read_text())
        self.assertEqual(source_checked(owner._forward_method),PATH.read_text())
    def test_regression_original_whole_ast_disagrees(self):
        original,owner=fixture()
        inspected=ast.parse(textwrap.dedent(inspect.getsource(owner._forward_method))).body[0]
        self.assertNotEqual(ast.dump(original),ast.dump(inspected))
        sliced=''.join(PATH.read_text().splitlines(keepends=True)[original.lineno-1:original.end_lineno])
        self.assertEqual(ast.dump(ast.parse(textwrap.dedent(sliced)).body[0]),ast.dump(inspected))
    def test_module_qualname_signature_not_relaxed(self):
        for attr,value in [('__module__','other'),('__qualname__','Other.forward_cuda'),('__defaults__',(None,))]:
            _,owner=fixture();fn=owner._forward_method.__func__;setattr(fn,attr,value)
            with self.assertRaises(RuntimeError):source_checked(owner._forward_method)
    def test_filename_and_line_not_relaxed(self):
        for change in ({'co_filename':'/wrong/file.py'},{'co_firstlineno':1}):
            _,owner=fixture();fn=owner._forward_method.__func__;fn.__code__=fn.__code__.replace(**change)
            with self.assertRaises((RuntimeError,OSError,IndentationError,SyntaxError)):
                source_checked(owner._forward_method)
    def test_file_sha_not_relaxed(self):
        _,owner=fixture();original=Path.read_bytes
        def read(path):return b'changed source' if path==PATH else original(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaisesRegex(RuntimeError,'source pin'):source_checked(owner._forward_method)
if __name__=='__main__':unittest.main()
