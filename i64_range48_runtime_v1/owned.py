import hashlib,inspect,json,textwrap
from pathlib import Path
R=Path(__file__).resolve().parent

def prepare(original,site,callback):
 spec=json.loads((R/'SOURCE.json').read_text())[site]
 assert original.__code__.co_filename==spec['path'] and original.__code__.co_firstlineno==spec['firstlineno'] and not original.__code__.co_freevars
 assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['file_sha256']
 source=textwrap.dedent(inspect.getsource(original));expected=(R/(site+'.original.py.txt')).read_text()
 assert source==expected and hashlib.sha256(source.encode()).hexdigest()==spec['input_sha256']
 candidate=(R/(site+'.candidate.py.txt')).read_text();assert candidate==source.replace(spec['expression'],spec['replacement']) and source.count(spec['expression'])==1
 assert hashlib.sha256(candidate.encode()).hexdigest()==spec['output_sha256']
 ns=dict(original.__globals__);ns['_range_guard']=callback
 exec(compile(candidate,str(R/(site+'.candidate.py.txt')),'exec'),ns);body=ns[original.__name__]
 body.__defaults__=original.__defaults__;body.__kwdefaults__=original.__kwdefaults__
 return body,dict(site=site,input_source_sha256=spec['input_sha256'],output_source_sha256=spec['output_sha256'],source_path=str(R/(site+'.candidate.py.txt')))
