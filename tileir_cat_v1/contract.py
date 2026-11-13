"""Exact observed cat contracts and CPU index oracle. No torch/CUDA import."""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
import json
from math import prod
from pathlib import Path

ORIGIN_SHA256 = 'bd7aabc9e7d4115936a82b072ca4d931697f5efe3ed876f6649e274b1dfec8ac'
BLOCK = 256

class UnsupportedCat(ValueError):
    """Unsupported metadata; native fallback is allowed only before launch."""

@dataclass(frozen=True)
class Snapshot:
    meta: dict
    object_id: int
    storage_address: int

@dataclass(frozen=True)
class CatPlan:
    site: str
    variant: int
    contract: dict

    @property
    def output(self):
        return self.contract['output']

    @property
    def numel(self):
        return prod(self.output['shape'])

    @property
    def kernel_family(self):
        if self.site.startswith('cat_gdn_'): return 'gdn32'
        if self.site.startswith('cat_kv_'):
            return 'kv_initial' if self.contract['empty_inputs'] else 'kv_append'
        return {'cat_rope': 'rope', 'cat_initial_positions': 'initial_positions',
                'cat_positions': 'positions'}[self.site]

@lru_cache(maxsize=1)
def _catalog():
    from .pins import CONTRACT_SHA256
    raw = Path(__file__).with_name('OBSERVED-CAT-CONTRACTS.json').read_bytes()
    if sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise RuntimeError('frozen cat contract hash mismatch')
    data = json.loads(raw)
    if data['origin_sha256'] != ORIGIN_SHA256:
        raise RuntimeError('unexpected observed-contract origin')
    return data


def contracts():
    """Defensive copies of frozen observed metadata for CPU tests/review."""
    return json.loads(json.dumps(_catalog()))


def _overlap(a, b):
    an, bn = a.meta['storage_nbytes'], b.meta['storage_nbytes']
    return an > 0 and bn > 0 and max(a.storage_address, b.storage_address) < min(a.storage_address + an, b.storage_address + bn)


def _validate_snapshot(s):
    m = s.meta
    size = m['element_size']
    if type(s.object_id) is not int or type(s.storage_address) is not int:
        raise UnsupportedCat('invalid identity metadata')
    if s.storage_address < 0 or (prod(m['shape']) and (s.storage_address == 0 or s.storage_address % size)):
        raise UnsupportedCat('invalid storage pointer/alignment')
    last = m['storage_offset'] + sum((n - 1) * stride for n, stride in zip(m['shape'], m['stride']))
    if prod(m['shape']) and (last + 1) * size > m['storage_nbytes']:
        raise UnsupportedCat('source view exceeds storage')


def match_contract(site, snapshots, dim=0, *, out_is_none=True):
    """CPU-only, exact metadata and simultaneous alias admission, no writes."""
    if type(site) is not str or site not in _catalog()['sites']:
        raise UnsupportedCat('unobserved_site')
    if type(dim) is not int or not out_is_none:
        raise UnsupportedCat('dim_type_or_out')
    for variant, item in enumerate(_catalog()['sites'][site]['variants']):
        c = item['contract']
        if dim != c['dim'] or len(snapshots) != c['input_count']:
            continue
        if any(s.meta != m for s, m in zip(snapshots, c['inputs'])):
            continue
        for s in snapshots:
            _validate_snapshot(s)
        aliases = {(a['left'], a['right']): a for a in c['input_aliases']}
        for i, j in combinations(range(len(snapshots)), 2):
            a, b = snapshots[i], snapshots[j]
            same_object = a.object_id == b.object_id
            same_storage = a.storage_address == b.storage_address
            expected = aliases[i, j]
            if same_object != expected['same_object'] or same_storage != expected['same_storage']:
                raise UnsupportedCat('input_alias_mismatch')
            if not same_storage and _overlap(a, b):
                raise UnsupportedCat('distinct_input_storages_overlap')
        return CatPlan(site, variant, c)
    raise UnsupportedCat('unobserved_input_metadata')


def validate_output(plan, output, inputs):
    if output.meta != plan.output:
        raise UnsupportedCat('fresh_output_metadata_mismatch')
    _validate_snapshot(output)
    for source in inputs:
        if source.object_id == output.object_id or source.storage_address == output.storage_address or _overlap(source, output):
            raise UnsupportedCat('fresh_output_aliases_input')


def source_for_output(plan, linear):
    """Independent cat oracle: source index and source physical element offset."""
    shape = plan.output['shape']
    if type(linear) is not int or not 0 <= linear < prod(shape):
        raise ValueError('output index out of bounds')
    coord = [0] * len(shape)
    for d in range(len(shape) - 1, -1, -1):
        coord[d] = linear % shape[d]
        linear //= shape[d]
    axis = plan.contract['normalized_axis']
    remaining = coord[axis]
    for i, m in enumerate(plan.contract['inputs']):
        if not prod(m['shape']):
            continue
        if remaining < m['shape'][axis]:
            coord[axis] = remaining
            offset = m['storage_offset'] + sum(a*b for a,b in zip(coord,m['stride']))
            return i, offset
        remaining -= m['shape'][axis]
    raise AssertionError('contract does not cover output')
