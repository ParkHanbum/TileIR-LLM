import ast
import unittest
from pathlib import Path
P=Path(__file__).resolve().parent.parent
class Tests(unittest.TestCase):
    def test_no_diagnostic_payload_operations(self):
        for name in ('four.py','shadow.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            forbidden=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.Call):
                    f=ast.unparse(n.func)
                    if f.endswith(('.cpu','.numpy','.tobytes','.clone')) or f in ('expected','cpu_copy_oracle','bytes'):forbidden.append(f)
            self.assertEqual(forbidden,[])
    def test_four_numerical_dispatch_identical(self):
        old=ast.parse((P/'residual_four_lean26_runtime_v1/adapter.py').read_text())
        new=ast.parse((Path(__file__).parent/'four.py').read_text())
        names={'api.prepare_rope_positions','api.combine_sampled_and_draft_tokens','api.apply_logit_bias','ct.launch','api.validate_layout','api.validate_values','validate_metadata','validate_values'}
        def calls(tree):return sorted(ast.dump(n)for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)in names)
        self.assertEqual(calls(old),calls(new))
    def test_no_raw_success_claim(self):
        source=(Path(__file__).parent/'shadow.py').read_text()
        self.assertIn('raw_rechecked=False',source);self.assertIn('cpu_oracle_calls=0',source)
        self.assertNotIn('cpu_oracle_passes',source)
if __name__=='__main__':unittest.main()
