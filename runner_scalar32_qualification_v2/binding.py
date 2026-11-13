# SPDX-License-Identifier: Apache-2.0
# Narrow public-source transformation; not a global Tensor patch.
import ast, inspect, json, hashlib, textwrap
from pathlib import Path
EXPECTED = json.loads(Path(__file__).with_name("SOURCE.json").read_text())
TARGET = ast.dump(ast.parse("self.num_accepted_tokens_gpu[req_index].fill_(1)").body[0], include_attributes=False)
def checked_source(fn):
    if inspect.getsourcefile(fn) != EXPECTED["path"] or fn.__code__.co_firstlineno != EXPECTED["firstlineno"]:
        raise ValueError("source identity")
    if hashlib.sha256(Path(EXPECTED["path"]).read_bytes()).hexdigest() != EXPECTED["sha256"]:
        raise ValueError("source SHA")
    s=textwrap.dedent(inspect.getsource(fn))
    if ast.dump(ast.parse(s), include_attributes=False) != EXPECTED["method_ast"]:
        raise ValueError("source AST")
    return s

def build(owner, helper):
    original=owner.__dict__["add_request"]
    s=checked_source(original)
    node=ast.parse(s).body[0]
    hits=0
    for i, stmt in enumerate(node.body):
        if ast.dump(stmt, include_attributes=False)==TARGET:
            node.body[i]=ast.parse("_tileir_fill(self.num_accepted_tokens_gpu, req_index)").body[0]
            hits+=1
    if hits!=1: raise ValueError("replacement count")
    # Keep zero-argument super semantics via lexical __class__ cell.
    factory=ast.parse("def factory(__class__):\n def placeholder(): pass\n return add_request\n").body[0]
    factory.body[0]=node
    module=ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[]))
    ns=dict(original.__globals__); ns["_tileir_fill"]=helper
    exec(compile(module, "<public-runner-fill>", "exec", flags=__import__('__future__').annotations.compiler_flag),ns)
    replacement=ns["factory"](owner)
    replacement.__defaults__=original.__defaults__
    replacement.__kwdefaults__=original.__kwdefaults__
    replacement.__annotations__=original.__annotations__
    return original,replacement

class Binding:
    def __init__(self, owner, helper):
        self.owner=owner; self.original,self.replacement=build(owner,helper)
        self.installed=False
    def install(self):
        if self.installed or self.owner.__dict__["add_request"] is not self.original:
            raise RuntimeError("binding changed")
        self.owner.add_request=self.replacement; self.installed=True
    def close(self):
        if self.installed:
            if self.owner.__dict__["add_request"] is not self.replacement: raise RuntimeError("binding changed")
            self.owner.add_request=self.original; self.installed=False
