"""Owned instance-method copy. No global module/torch monkey patch."""
import ast,inspect,textwrap,types
from pathlib import Path
SOURCE=Path(__file__).resolve().parent.parent/'residual_four_gdn_owned_lean_v2/four.py'
def make_method(original,callback):
    tree=ast.parse(SOURCE.read_text());expected=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Replacement')
    expected=next(n for n in expected.body if isinstance(n,ast.FunctionDef)and n.name=='call')
    actual=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    if ast.dump(actual,include_attributes=False)!=ast.dump(expected,include_attributes=False):raise RuntimeError('Unexpected post-update caller source')
    class Rewrite(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name)and node.func.id=='validate_values':
                if [ast.dump(x,include_attributes=False)for x in node.args]!=[ast.dump(ast.Name(id=n,ctx=ast.Load()),include_attributes=False)for n in ('torch','args')]:raise RuntimeError('Unexpected predicate arguments')
                node.func.id='_post_update_value_guard';self.count+=1
            return node
    rewriter=Rewrite();node=rewriter.visit(actual);assert rewriter.count==1
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace=dict(original.__globals__);namespace['_post_update_value_guard']=callback
    exec(compile(module,str(SOURCE)+'.post_update_guard_copy','exec'),namespace)
    return namespace['call']
class Binding:
    def __init__(self,owner,callback):self.owner=owner;self.callback=callback;self.installed=False
    def install(self):
        if self.installed or 'call'in vars(self.owner):raise RuntimeError('Existing owned override')
        self.original=self.owner.call;self.copied=types.MethodType(make_method(self.original,self.callback),self.owner)
        self.owner.call=self.copied;self.installed=True
    def close(self):
        if not self.installed:return
        changed=self.owner.call is not self.copied
        del self.owner.call;self.installed=False
        if changed:raise RuntimeError('Owned post-update method replaced concurrently')
