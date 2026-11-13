"""Static ORIGINAL-v7 GEMM lazy-import closure; never imports tensor backends.

Package __init__ import edges are included before each requested child module.
A local module string is an edge even inside a lazy function/dispatch table.
This proves tracked source closure, not execution of CUDA or model code.
"""
import ast
from collections import deque
from pathlib import Path
import unittest
from regression_v7.common import REPO, tracked_map, check_file
ROOTS=('tileirblas_once_occ2.api','tileirblas_once_occ2._enqueue')
LAZY_KERNELS={'tileirblas_decode.kernels','tileirblas_prefill.kernels','tileirblas_blocked.fallback','tileirblas_blocked.kernels'}

def local_paths(module,root=REPO):
    parts=module.split('.');out=[]
    if not all(part.isidentifier() for part in parts):return out
    for end in range(1,len(parts)+1):
        rel=Path(*parts[:end])/'__init__.py'
        if (root/rel).is_file():out.append(str(rel))
    rel=Path(*parts).with_suffix('.py')
    if (root/rel).is_file():out.append(str(rel))
    return out

def closure(roots=ROOTS,root=REPO):
    queue=deque((name,'entry') for name in roots);seen=set();edges={};dynamic=[]
    while queue:
        module,origin=queue.popleft()
        for rel in local_paths(module,root):
            edges.setdefault(rel,set()).add(origin)
            if rel in seen:continue
            seen.add(rel);tree=ast.parse((root/rel).read_text())
            package=rel[:-12].replace('/','.') if rel.endswith('/__init__.py') else rel.rsplit('/',1)[0].replace('/','.')
            for node in ast.walk(tree):
                modules=[]
                if isinstance(node,ast.Import):modules=[x.name for x in node.names]
                elif isinstance(node,ast.ImportFrom):
                    base=node.module or ''
                    if node.level:base='.'.join(package.split('.')[:len(package.split('.'))-node.level+1])+('.'+base if base else '')
                    modules=[base]+[base+'.'+x.name for x in node.names if x.name!='*']
                elif isinstance(node,ast.Constant) and isinstance(node.value,str) and '.' in node.value and all(p.isidentifier() for p in node.value.split('.')):
                    modules=[node.value] if local_paths(node.value,root) else []
                if isinstance(node,ast.Call) and ((isinstance(node.func,ast.Name) and node.func.id in ('import_module','__import__')) or (isinstance(node.func,ast.Attribute) and node.func.attr=='import_module')):
                    dynamic.append(dict(path=rel,line=node.lineno,expression=ast.unparse(node)))
                for name in modules:queue.append((name,rel+':'+str(node.lineno)))
    return dict(files=sorted(seen),edges={p:sorted(e) for p,e in edges.items()},dynamic=dynamic)

class LazyImports(unittest.TestCase):
    def test_all_four_lazy_dispatch_modules_and_initializers(self):
        path=REPO/'tileirblas_once_occ2/_enqueue.py'
        strings={n.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertEqual({s for s in strings if s.startswith('tileirblas_')},LAZY_KERNELS)
        result=closure();files=set(result['files'])
        self.assertIn('tileirblas_decode/__init__.py',files)
        self.assertIn('tileirblas_decode/api.py',files)
        self.assertIn('tileirblas_tuned/__init__.py',files)
        self.assertIn('tileirblas_tuned/api.py',files)
        for name in LAZY_KERNELS:self.assertTrue(set(local_paths(name))<=files)
        self.assertEqual([(x['path'],x['expression']) for x in result['dynamic']],[('tileirblas_once_occ2/_enqueue.py','import_module(module)')])
    def test_every_local_dependency_is_tracked_and_head_exact(self):
        tracked=tracked_map()
        for rel in closure()['files']:check_file(REPO/rel,tracked)
    def test_ignored_transitive_initializer_is_rejected(self):
        tracked=tracked_map();tracked.pop('tileirblas_tuned/__init__.py',None)
        with self.assertRaises(AssertionError):check_file(REPO/'tileirblas_tuned/__init__.py',tracked)
if __name__=='__main__':unittest.main()
