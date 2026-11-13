"""Composable owned-forward AST edit, not an installer or global monkeypatch.
Caller supplies already isolated forward AST; preserve other edits/core binding.
Only exact three source expressions are replaced, retaining evaluation order.
"""
import ast
import copy

def transform_forward(node):
    node=copy.deepcopy(node)
    counts=[0,0,0]
    # Only main-branch top-level assignment; leave fused-spec-decode branch alone.
    candidates=[n for n in node.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="core_attn_out"]
    if len(candidates)!=1: raise ValueError("main branch assignment census")
    target=candidates[0]
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,n):
            if n is target and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='core_attn_out' and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.zeros':
                counts[0]+=1
                n.value.func=ast.Attribute(ast.Name('_owned_gdn_copies',ast.Load()),'zeros',ast.Load())
            return self.generic_visit(n)
        def visit_Call(self,n):
            if ast.unparse(n.func)=='torch.ops.vllm.qwen_gdn_attention_core':
                if len(n.args)<4 or ast.unparse(n.args[1])!='b.contiguous()' or ast.unparse(n.args[2])!='a.contiguous()':
                    raise ValueError('unexpected core argument contract')
                for i,name in ((1,'b'),(2,'a')):
                    n.args[i]=ast.Call(ast.Attribute(ast.Name('_owned_gdn_copies',ast.Load()),'contiguous_'+name,ast.Load()),[ast.Name(name,ast.Load())],[])
                    counts[i]+=1
            return self.generic_visit(n)
    result=Rewrite().visit(node)
    if counts!=[1,1,1]: raise ValueError('owned-forward source census mismatch')
    return ast.fix_missing_locations(result)
