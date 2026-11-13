import inspect,ast,textwrap,json,hashlib
from pathlib import Path
def verify(runner):
 d=json.loads(Path(__file__).with_name('CALLER.json').read_text());fn=runner.add_requests
 assert 'add_requests' not in runner.__dict__ and fn.__func__ is type(runner).__dict__['add_requests']
 assert inspect.getsourcefile(fn)==d['path'] and fn.__code__.co_firstlineno==d['line']
 assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
 assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))),include_attributes=False)==d['ast']
 return {'dynamic_model_state_call_source_verified':True,'actual_target_census_pending':True}
