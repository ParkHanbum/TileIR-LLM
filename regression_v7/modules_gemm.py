"""Three saved real v7 GEMMs. Explicit GPU entry only; no timing or model claim."""
import hashlib
import json
import math
import struct
from pathlib import Path

P = Path(__file__).resolve().parents[1]
MANIFEST = P / 'packed_k128_qualification_v1/MANIFEST.json'
MANIFEST_SHA = '28245ce86f8ccef6b55bcce0c6ebca220112a582a8879a1456ebc1295d73ef52'
CASES = ('ba_decode_layer1_mismatch', 'qkvz_decode', 'gate_up_prefill')
SHAPES = ((32, 64, 2560), (32, 12288, 2560), (1696, 18432, 2560))
BLAS_SHA = '25cde432108543626df85d887c730bf2cab96803ca16f9654e4abc8d2b45833e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows():
    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('GEMM manifest changed')
    m = json.loads(MANIFEST.read_text())
    from packed_k128_qualification_v1.run import validate_manifest
    all_rows = validate_manifest(m)
    selected = [next(r for r in all_rows if r['label'] == name) for name in CASES]
    for r, (a, b, k) in zip(selected, SHAPES):
        for name, shape in (('X', [a, k]), ('W', [b, k])):
            meta = r['metadata'][name]
            if meta['shape'] != shape or meta['stride'] != [k, 1] or meta['storage_offset'] != 0:
                raise RuntimeError('Unsupported saved GEMM layout')
    return selected


def dependencies():
    from integration.direct_reference_v4 import BLAS
    result = {str(MANIFEST): MANIFEST_SHA, str(BLAS): BLAS_SHA,
              str(P / 'reference/libpublic_cublas.so'): '987b0c98a37196b4c49677471625b1bed1c1c963522978f9a647651f1d9a1366'}
    for r in rows():
        for key in ('X', 'W'):
            result[r[key]['path']] = r[key]['sha256']
    return result


def cpu_tests():
    # Run each module in a fresh CPU process: two suites assert no torch import.
    return ['tileirblas_once_occ2.tests.test_contract',
            'tileirblas_once_occ2.tests.test_hint',
            'tileirblas_blocked.tests.test_contract',
            'tileirblas_blocked.tests.test_block_mapping',
            'packed_k128_partial_v1.test_cpu',
            'packed_k128_qualification_v1.test_cpu', 'regression_v7.test_gemm']


def require_comparison(c):
    required = ('raw_equal', 'raw_mismatches', 'strict_failures',
                'reference_nonfinite', 'candidate_nonfinite')
    if any(k not in c for k in required) or c['raw_equal'] is not True or any(c[k] != 0 for k in required[1:]):
        raise AssertionError('GEMM raw/strict/finite regression')


def compare_small_words(reference, candidate):
    """Bounded CPU detector fixtures, BF16 integer words; not a GPU oracle."""
    if len(reference) != len(candidate) or not reference or len(reference) > 32:
        raise ValueError('Expected equal nonempty fixtures of at most 32 words')
    if any(type(w) is not int or not 0 <= w <= 65535 for w in (*reference, *candidate)):
        raise ValueError('Invalid BF16 word')
    decode = lambda w: struct.unpack('<f', struct.pack('<I', w << 16))[0]
    a, b = list(map(decode, reference)), list(map(decode, candidate))
    mismatches = sum(x != y for x, y in zip(reference, candidate))
    return dict(raw_equal=mismatches == 0, raw_mismatches=mismatches,
                strict_failures=sum(not(math.isfinite(x) and math.isfinite(y)) or abs(x-y) > 1e-5 + 1e-5*abs(x) for x,y in zip(a,b)),
                reference_nonfinite=sum(not math.isfinite(x) for x in a),
                candidate_nonfinite=sum(not math.isfinite(x) for x in b))


def run_gpu(torch, output):
    """Caller owns bounded process/lock/cache. Raises on failure; never retries."""
    from packed_k128_qualification_v1.run import direct_default99, compare
    from integration.direct_reference_v4 import tensor_sha
    from packed_k128_partial_v1 import linear as packed
    from tileirblas_once_occ2 import linear as base
    from tileirblas_once_occ2 import api as base_api
    for path, digest in dependencies().items():
        if sha(path) != digest:
            raise RuntimeError('External GEMM dependency changed: ' + path)
    flags = lambda: [torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
    if flags() != [True, True]:
        raise RuntimeError('GEMM requires unchanged T/T precision flags')
    device = torch.cuda.current_device()
    prior_stream = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    reference = None
    retained = []
    result = dict(complete=False, timing=False, cases=[], reference_version=None,
                  reference='fresh public GemmEx D99/math0/compute68; never historical F',
                  selected_inputs='saved real matrices; not all 2448 model calls')
    try:
        with torch.cuda.device(device), torch.cuda.stream(stream):
            reference, native, lib = direct_default99(torch)
            if reference.version != 130101 or sha(lib) != BLAS_SHA:
                raise RuntimeError('Reference version/library changed')
            result['reference_version'] = reference.version
            for row, shape in zip(rows(), SHAPES):
                m, n, k = shape
                cpu = {}
                for key, expected in (('X', (m, k)), ('W', (n, k))):
                    t = torch.load(row[key]['path'], map_location='cpu', weights_only=True)
                    if t.dtype != torch.bfloat16 or tuple(t.shape) != expected or not t.is_contiguous():
                        raise RuntimeError('Saved tensor differs from manifest')
                    cpu[key] = t
                # Full owning allocations, no unobserved prefix/suffix view bytes.
                x = cpu['X'].to(device=device); w = cpu['W'].to(device=device)
                retained[:] = [x, w]
                for t in (x, w):
                    if t.storage_offset() != 0 or t.untyped_storage().nbytes() != t.numel() * 2:
                        raise RuntimeError('Unexpected backing extent')
                stream.synchronize()
                before = (tensor_sha(x), tensor_sha(w))
                refs = []; native_records = []
                for repeat in range(2):
                    y = torch.empty((m, n), dtype=torch.bfloat16, device=device)
                    retained.append(y)
                    meta = native(x, w, y)
                    if meta['stream'] != stream.cuda_stream:
                        raise RuntimeError('Reference stream mismatch')
                    native_records.append({key: value for key, value in meta.items() if key != 'stream'})
                    stream.synchronize(); refs.append(y.cpu())
                repeated = compare(refs[0], refs[1], torch); require_comparison(repeated)
                comparisons = []
                fn = packed if row['label'] == CASES[0] else base
                for repeat in range(2):
                    y = fn(x, w); retained.append(y)
                    if y.dtype != torch.bfloat16 or tuple(y.shape) != (m, n) or y.device != x.device or not y.is_contiguous() or y.storage_offset() != 0:
                        raise AssertionError('GEMM output contract')
                    if y.untyped_storage().data_ptr() in (x.untyped_storage().data_ptr(), w.untyped_storage().data_ptr()):
                        raise AssertionError('GEMM output alias')
                    if torch.cuda.current_stream(device) != stream:
                        raise AssertionError('Current stream changed')
                    stream.synchronize(); value = y.cpu()
                    comparison = compare(refs[0], value, torch); require_comparison(comparison)
                    if (tensor_sha(x), tensor_sha(w)) != before:
                        raise AssertionError('GEMM input/full backing modified')
                    comparisons.append(comparison)
                # Deliberate corruption of independent CPU returned-output copies only.
                bit_bad = value.clone(); bits = bit_bad.view(torch.int16).flatten(); bits[0] ^= 1
                bit_result = compare(refs[0], bit_bad, torch)
                try: require_comparison(bit_result)
                except AssertionError: pass
                else: raise AssertionError('Raw detector accepted deliberate bit regression')
                large_bad = value.clone(); large_bad.flatten()[0] = -32768.0 if float(value.flatten()[0]) >= 0 else 32768.0
                large_result = compare(refs[0], large_bad, torch)
                if large_result['strict_failures'] < 1 or large_result['candidate_nonfinite'] != 0:
                    raise AssertionError('Strict detector failed large finite regression')
                result['cases'].append(dict(label=row['label'],mnk=list(shape), backend='packed_k128_partial_v1' if fn is packed else 'tileirblas_once_occ2',
                    base_path=None if fn is packed else base_api.selected_path(*shape), native_calls=native_records,
                    native_repeat=repeated, candidate_comparisons=comparisons,
                    input_full_backing_readonly=True, output_nonalias=True, nondefault_current_stream=True,
                    raw_negative_detected=True, strict_finite_negative_detected=True,
                    raw_negative=bit_result, strict_negative=large_result))
                stream.synchronize(); retained.clear()
            if flags() != [True, True]: raise AssertionError('Precision flags changed')
    finally:
        # Synchronize before releasing pending tensor refs or destroying native handle.
        # A sync failure propagates; no success receipt or retry is manufactured.
        stream.synchronize()
        if reference is not None: reference.close()
        retained.clear()
    if torch.cuda.current_device() != device or torch.cuda.current_stream(device) != prior_stream:
        raise AssertionError('Device/stream context not restored')
    result.update(complete=True, context_restored=True, reference_closed=True, precision_flags=[True,True], gpu_candidate_calls=6, gpu_reference_calls=6)
    audit(result)
    return result


def audit(result):
    """Standalone stored-result gate. Missing/duplicate/partial cases fail closed."""
    def require(ok, message):
        if not ok: raise AssertionError('GEMM audit: ' + message)
    require(type(result) is dict, 'result object')
    require(all(result.get(k) is True for k in ('complete','context_restored','reference_closed')), 'completion/lifetime')
    require(result.get('timing') is False and not result.get('error') and not result.get('cleanup_errors'), 'errors/timing')
    require(result.get('reference_version') == 130101, 'reference version')
    require(result.get('reference') == 'fresh public GemmEx D99/math0/compute68; never historical F', 'reference contract')
    require(result.get('precision_flags') == [True, True], 'precision flags')
    require(type(result.get('gpu_candidate_calls')) is int and result['gpu_candidate_calls'] == 6 and type(result.get('gpu_reference_calls')) is int and result['gpu_reference_calls'] == 6, 'total call counts')
    cases = result.get('cases')
    require(type(cases) is list and len(cases) == 3, 'case count')
    require([c.get('label') for c in cases] == list(CASES), 'case order/identity')
    reference = dict(math_mode=0,pointer_mode=0,algo=99,compute_type=68,alpha=1,beta=0)
    for i, (case, shape) in enumerate(zip(cases, SHAPES)):
        require(case.get('mnk') == list(shape), 'shape')
        require(case.get('backend') == ('packed_k128_partial_v1' if i == 0 else 'tileirblas_once_occ2'), 'backend')
        require(case.get('base_path') == (None if i == 0 else 'blocked' if i == 1 else 'prefill128'), 'base path')
        require(all(case.get(k) is True for k in ('input_full_backing_readonly','output_nonalias','nondefault_current_stream','raw_negative_detected','strict_finite_negative_detected')), 'input/output/stream/negative contract')
        require(case.get('native_calls') == [reference,reference], 'two fresh native calls')
        comparisons = case.get('candidate_comparisons')
        require(type(comparisons) is list and len(comparisons) == 2, 'two candidate calls')
        for c in [case.get('native_repeat',{}), *comparisons]:
            require_comparison(c)
            require(c.get('elements') == shape[0]*shape[1] and c.get('atol') == 1e-5 and c.get('rtol') == 1e-5, 'comparison size/tolerance')
        for kind in ('raw_negative','strict_negative'):
            c=case.get(kind,{})
            require(c.get('raw_equal') is False and c.get('raw_mismatches') == 1, 'negative raw detection')
            require(c.get('elements') == shape[0]*shape[1] and c.get('atol') == 1e-5 and c.get('rtol') == 1e-5, 'negative comparison scope')
            require(c.get('reference_nonfinite') == 0 and c.get('candidate_nonfinite') == 0, 'negative finite contract')
            if kind == 'strict_negative': require(type(c.get('strict_failures')) is int and c['strict_failures'] >= 1, 'negative strict detection')
    return dict(passed=True, cases=3, candidate_calls=6, reference_calls=6, raw_strict_finite=True, lifetime_contract=True, negative_detectors=True)
