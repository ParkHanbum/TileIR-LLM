import ast,json,copy,types,unittest
from pathlib import Path
from .source_gate import verify
from .cleanup import cleanup
class Tests(unittest.TestCase):
 def test_actual_filename_lineno_inspect_path(self):
  root=Path(__file__).parent;mods={'buffer':types.SimpleNamespace(),'block':types.SimpleNamespace()}
  for row in json.loads((root/'METHODS.json').read_text()):
   tree=ast.parse(Path(row['path']).read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==row['owner']);fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name==row['attribute']))
   fn.decorator_list=[];fn.returns=None
   for arg in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs:arg.annotation=None
   fn.args.defaults=[ast.Constant(None)for _ in fn.args.defaults];fn.args.kw_defaults=[ast.Constant(None)if x else None for x in fn.args.kw_defaults]
   fn.body=[ast.copy_location(ast.Pass(),fn.body[0])];module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));ns={};exec(compile(module,row['path'],'exec'),ns)
   owner=getattr(mods[row['module']],row['owner'],None)
   if owner is None:owner=type(row['owner'],(),{});setattr(mods[row['module']],row['owner'],owner)
   setattr(owner,row['attribute'],ns[row['attribute']])
  self.assertEqual(len(verify(mods['buffer'],mods['block'],root)),4)
 def test_cleanup_failure_still_shutdown_no_release(self):
  events=[]
  def bad():events.append('restore');raise ValueError()
  o=types.SimpleNamespace(close=bad,release_after_sync=lambda:events.append('release'),safe_receipt=lambda:{})
  engine=types.SimpleNamespace(engine_core=types.SimpleNamespace(shutdown=lambda:events.append('shutdown')))
  with self.assertRaises(RuntimeError):cleanup({},lambda:None,None,engine,o,lambda:events.append('sync'),lambda:None,False)
  self.assertEqual(events,['restore','sync','shutdown'])
 def test_source_allowlist_no_stale_receipt(self):
  r=Path(__file__).parent
  self.assertFalse((r/'CACHE-PRESEED-01.json').exists());self.assertFalse((r/'cache_v1').exists())
