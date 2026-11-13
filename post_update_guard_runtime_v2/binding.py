"""Model-local restoration fix; install/source-copy and numeric APIs unchanged."""
from public_post_update_guard_v1.binding import Binding as OriginalBinding
class Binding(OriginalBinding):
 def close(self):
  if not self.installed:return
  # Inspect the instance slot, not a class fallback or foreign descriptor.
  if vars(self.owner).get('call') is not self.copied:
   raise RuntimeError('Owned post-update method replaced concurrently')
  del self.owner.call
  self.installed=False
