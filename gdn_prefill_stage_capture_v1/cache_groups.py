"""Redirect copied Triton group indexes to verified copied siblings, never old absolute paths."""
from pathlib import Path
def relocate(group,directory,filenames):
 assert type(group) is dict and set(group)=={'child_paths'}
 children=group['child_paths'];assert type(children) is dict
 assert set(children)==set(filenames)
 assert all(type(k) is str and Path(k).name==k and k not in ('.','..') for k in children)
 return {'child_paths':{k:str(Path(directory)/k) for k in children}}
