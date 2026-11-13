import hashlib,inspect,json,textwrap
from pathlib import Path
R=Path(__file__).resolve().parent

def prepare(original,site,callback):
 spec=json.loads((R/'RANGE-SOURCE.json').read_text())[site]
 assert original.__code__.co_filename==spec['path'] and original.__code__.co_firstlineno==spec['firstlineno'] and not original.__code__.co_freevars
 assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['file_sha256']
 source=validated_source(original,spec);expected=(R/('range_'+site+'.original.py.txt')).read_text()
 assert source==expected and hashlib.sha256(source.encode()).hexdigest()==spec['input_sha256']
 candidate=(R/('range_'+site+'.candidate.py.txt')).read_text();assert candidate==source.replace(spec['expression'],spec['replacement']) and source.count(spec['expression'])==1
 assert hashlib.sha256(candidate.encode()).hexdigest()==spec['output_sha256']
 ns=dict(original.__globals__);ns['_range_guard']=callback
 exec(compile(candidate,str(R/('range_'+site+'.candidate.py.txt')),'exec'),ns);body=ns[original.__name__]
 body.__defaults__=original.__defaults__;body.__kwdefaults__=original.__kwdefaults__
 return body,dict(site=site,input_source_sha256=spec['input_sha256'],output_source_sha256=spec['output_sha256'],source_path=str(R/('range_'+site+'.candidate.py.txt')))

def validated_source(original,spec):
 if 'compile_prefix' not in spec:return textwrap.dedent(inspect.getsource(original))
 source=Path(spec['path']).read_text()
 # Reconstruct EXACT production factory compile input, including future prefix.
 # Equality of full CodeType recursively checks constants/nested code, flags,
 # bytecode, filename and source positions. No path/firstline relaxation.
 compiled=compile(spec['compile_prefix']+source,spec['path'],'exec',dont_inherit=True)
 codes=[c for c in compiled.co_consts if isinstance(c,type(original.__code__)) and c.co_name==original.__name__]
 assert len(codes)==1 and original.__code__==codes[0],'Production finite code provenance mismatch'
 return source
