"""Explicitly balanced schedules; no accidental random all-native-first sequence."""
BASE=(('native','lean'),('lean','native'),('lean','native'),('native','lean'))
def orders(index):
 assert index in(0,1)
 return [list(p if index==0 else tuple(reversed(p)))for p in BASE]
def profile_order(index):
 assert index in(0,1)
 return ['native','lean']if index==0 else['lean','native']
def validate_schedule(index,pairs):
 assert len(pairs)==4
 assert [p['order']for p in pairs]==orders(index)
 for p in pairs:assert[p['requests'][i]['mode']for i in range(2)]==p['order']
 return True
