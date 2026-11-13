import json
from pathlib import Path
WORDS=[int(s,16)for s in json.loads((Path(__file__).parent.parent/'public_eq_all_guard_v2/SYNTHETIC-CASES.json').read_text())['input_words_hex']]
def cases():
 rows=[]
 for constant in (0,1):
  for kind,n,offset in [('scalar',1,0),('broadcast',128,0),('broadcast',128,1),('dense',1025,0)]:
   for word in WORDS:
    rows.append(dict(kind=kind,n=n,offset=offset,constant=constant,word=word,index=1024 if n==1025 else 0))
  for n in (0,1023,1024,1<<24):
   rows.append(dict(kind='dense',n=n,offset=0,constant=constant,word=0 if constant==0 else 0x3f800000,index=0))
  for pos in (0,(1<<23),(1<<24)-1):
   rows.append(dict(kind='dense',n=1<<24,offset=0,constant=constant,word=1 if constant==0 else 0x3f800001,index=pos))
 return [dict(id=i,**r)for i,r in enumerate(rows)]
def expected(row):
 if row['n']==0:return True
 return (row['word']&0x7fffffff)==0 if row['constant']==0 else row['word']==0x3f800000
