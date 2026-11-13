"""EXTERNAL DRAFT: original-v7 public attention/KV replay, no launcher/cache setup."""
from pathlib import Path
from types import SimpleNamespace
import json, hashlib
P = Path('/home/n/work/TileIR/public-tileir-blas-20261111')
ATT_MANIFEST = Path('/home/n/work/TileIR/public-tileir-blas-20261111/paged_attention_qualification_v10/MANIFEST.json')
KV_MANIFEST = Path('/home/n/work/TileIR/public-tileir-blas-20261111/attention_kv_runtime_capture_v1/attentionKVCapture01/bank/MANIFEST.json')
ROOT_PINS = {'/home/n/work/TileIR/public-tileir-blas-20261111/paged_attention_qualification_v10/MANIFEST.json': '3b89242bb81b2b91f4c9c6d7e0413179858a429673a172ecd133f2ca150a14a2', '/home/n/work/TileIR/public-tileir-blas-20261111/attention_kv_runtime_capture_v1/attentionKVCapture01/bank/MANIFEST.json': '05803a7c7f47e2f9c5df0fe4f3723ebadc3c0a97f85dad59d6c09e24b73c69bd', '/home/n/work/TileIR/public-tileir-blas-20261111/paged_attention_qualification_v10/run.py': '1794945ce57bbcd171e181d4c2ae845056d1e3bf77b416fbfddbff10705742f2', '/home/n/work/TileIR/public-tileir-blas-20261111/attention_kv_qualification_v2/run.py': 'fc8ea9c12b95552458f819a19dfe74407bb4cef48fb1ac4b4feecee917b20d41'}
from packed_grouped_qualification_v1.contract import sha
from gdn_post_conv_qualification_v2.run import compare as original_compare
from paged_attention_qualification_v10.run import validate_manifest as attention_manifest
from attention_kv_qualification_v2.run import validate_manifest as kv_manifest, selected_gate

ATT_SHAPES=[[32,16,256],[32,16,256],[1696,16,256],[1696,16,256],[1696,16,256],[32,16,256]]
KV_SHAPES={'prefill':[1696,4,256],'decode':[32,4,256]}
def compare(ref,out,torch):
 result=original_compare(ref,out,torch)
 result.update(shape=list(out.shape),dtype=str(out.dtype),elements=out.numel())
 return result
def metric_pass(m,shape):
 import math
 return (m.get('raw_equal')is True and all(type(m.get(k))is int and m[k]==0 for k in ('raw_different_elements','strict_failures','candidate_nonfinite','reference_nonfinite')) and m.get('strict_atol')==1e-5 and m.get('strict_rtol')==1e-5 and m.get('shape')==shape and m.get('dtype')=='torch.bfloat16' and type(m.get('elements'))is int and m['elements']==math.prod(shape))

def cpu_tests():
    return ['paged_attention_qualification_v10.test_cpu', 'attention_kv_qualification_v2.test_cpu', 'regression_v7.test_attention_kv']

def dependencies():
    refs = dict(ROOT_PINS)

    def walk(v):
        if isinstance(v, dict):
            if 'path' in v and 'sha256' in v:
                refs[v['path']] = v['sha256']
                if str(v['path']).endswith('.json'):
                    p = Path(v['path'])
                    assert sha(p) == v['sha256']
                    walk(json.loads(p.read_text()))
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for p in (ATT_MANIFEST, KV_MANIFEST):
        assert sha(p) == ROOT_PINS[str(p)]
        walk(json.loads(p.read_text()))
    return refs

def audit(result):
    try:
        assert result['complete'] is True
        a, k = (result['attention'], result['kv'])
        assert a['complete'] is k['complete'] is True
        expected_a = [(i, kind, j) for i in range(6) for kind in ('native', 'tile') for j in range(2)]
        assert [(x['bank_index'], x['kind'], x['repeat_index']) for x in a['records']] == expected_a
        for x in a['records']:
            assert all((x[n] is True for n in ('passed', 'inputs_preserved', 'return_out_alias', 'repeat_raw', 'same_stream', 'output_disjoint')))
            m = x['metrics']
            assert metric_pass(m,ATT_SHAPES[x['bank_index']])
        expected_k = [(phase, arm, j) for phase in ('prefill', 'decode') for arm in ('native', 'tile') for j in range(2)]
        assert [(x['phase'], x['arm'], x['repeat']) for x in k['records']] == expected_k
        for x in k['records']:
            assert all((x[n] is True for n in ('passed', 'return_none', 'full_owned_storage_raw_equal', 'input_full_owned_storage_preserved', 'repeat_raw_equal', 'shared_backing', 'same_stream')))
            assert selected_gate(x['selected_metrics'])
            assert set(x['selected_metrics'])=={'key','value'} and all(metric_pass(m,KV_SHAPES[x['phase']])for m in x['selected_metrics'].values())
        return True
    except (AssertionError, KeyError, TypeError):
        return False

def _attention(torch, output):
    samples = attention_manifest(json.loads(ATT_MANIFEST.read_text()))
    a = SimpleNamespace(output=output)
    output.mkdir(exist_ok=False)
    r = dict(complete=False, records=[], scope='Six saved logical banks; sentinel reconstruction, not live whole engine owner')

    def save():
        (output / 'RESULT.json').write_text(json.dumps(r, indent=2) + '\n')
    from vllm.vllm_flash_attn import flash_attn_varlen_func as native
    from tileir_paged_attention_v10 import attention

    def digest(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def load(e):
        assert sha(e['path']) == e['sha256']
        t = torch.load(e['path'], map_location='cpu', weights_only=True)
        assert digest(t) == e['logical_sha256']
        return t
    with torch.inference_mode():
        for bank_index, s in enumerate(samples):
            q = load(s['q']).cuda()
            target = load(s['output']).cuda()
            vk = load(s['valid_k'])
            vv = load(s['valid_v'])
            md = {}
            for key, e in s['metadata'].items():
                t = load(e)
                d = e['metadata']
                if d['stride'] == [0, 0]:
                    assert bool((t == 1).all())
                    md[key] = t[0, 0].reshape(1).cuda().as_strided(d['shape'], d['stride'], d['storage_offset'])
                else:
                    assert tuple(t.stride()) == tuple(d['stride']) and d['storage_offset'] == 0
                    md[key] = t.cuda()
            kd, vd = (s['tensors']['k'], s['tensors']['v'])
            shape = kd['shape']
            stride = kd['stride']
            ko = kd['storage_offset']
            vo = vd['storage_offset']
            assert vo == ko + 256 and stride == vd['stride']
            from attention_layer11_stages_diagnostic_v1.run import checked_extent
            extent = checked_extent(kd, vd)
            backing = torch.full((extent,), -37, dtype=torch.bfloat16, device='cuda')
            k = backing.as_strided(shape, stride, ko)
            v = backing.as_strided(shape, stride, vo)
            cursor = 0
            used = set()
            table = md['block_table'].cpu()
            lengths = md['seqused_k'].cpu().tolist()
            for b, L in enumerate(lengths):
                for pos in range(L):
                    block = int(table[b, pos // shape[1]])
                    off = pos % shape[1]
                    assert (block, off) not in used
                    used.add((block, off))
                    k[block, off].copy_(vk[cursor + pos])
                    v[block, off].copy_(vv[cursor + pos])
                cursor += L
            assert cursor == len(vk) == len(vv)
            inputs = dict(q=q, backing=backing, **md)
            before = {n: digest(t) for n, t in inputs.items()}
            native_ref = None
            tile_ref = None
            for kind in ('native', 'native', 'tile', 'tile'):
                out = torch.full_like(q, float('nan'))
                stream_before = torch.cuda.current_stream(q.device).cuda_stream
                out_range = (out.untyped_storage().data_ptr(), out.untyped_storage().data_ptr() + out.untyped_storage().nbytes())
                disjoint = all((out_range[1] <= t.untyped_storage().data_ptr() or t.untyped_storage().data_ptr() + t.untyped_storage().nbytes() <= out_range[0] for t in inputs.values()))
                assert disjoint
                if kind == 'native':
                    ret = native(q=q, k=k, v=v, out=out, **md, **s['scalar_args'])
                else:
                    ret = attention(q, k, v, md['cu_seqlens_q'], md['seqused_k'], md['block_table'], out)
                torch.cuda.synchronize()
                assert isinstance(ret, torch.Tensor)
                alias = ret.untyped_storage().data_ptr() == out.untyped_storage().data_ptr() and ret.shape == out.shape and (ret.stride() == out.stride())
                metric = compare(target, out, torch)
                preserved = all((digest(t) == before[n] for n, t in inputs.items()))
                ref = native_ref if kind == 'native' else tile_ref
                repeat = True if ref is None else torch.equal(ref.view(torch.uint8), out.view(torch.uint8))
                passed = metric['raw_equal'] and metric['strict_failures'] == 0 and (metric['candidate_nonfinite'] == metric['reference_nonfinite'] == 0) and preserved and alias and repeat
                row = dict(same_stream=torch.cuda.current_stream(q.device).cuda_stream == stream_before, output_disjoint=disjoint, repeat_index=len(r['records']) % 2, bank_index=bank_index, module=s['module'], phase=s['phase'], kind=kind, metrics=metric, inputs_preserved=preserved, return_out_alias=alias, repeat_raw=repeat, passed=passed)
                r['records'].append(row)
                assert row['same_stream']
                if not passed:
                    torch.save(out.cpu(), a.output / (str(bank_index) + '_' + s['phase'] + '_' + kind + '_difference.pt'))
                    r['raw_difference_hold'] = True
                    save()
                    raise AssertionError('Raw replay HOLD')
                if kind == 'native':
                    native_ref = out.clone()
                else:
                    tile_ref = out.clone()
                save()
            del inputs, backing, k, v, q, target, vk, vv, md, native_ref, tile_ref, out, ret
        r['complete'] = True
        save()
        return r

def _kv(torch, output):
    samples = kv_manifest(json.loads(KV_MANIFEST.read_text()))
    a = SimpleNamespace(output=output)
    output.mkdir(exist_ok=False)
    r = dict(complete=False, records=[], scope='Actual input values/slots; compact-prefix sentinel cache, not full engine backing')

    def save():
        (output / 'RESULT.json').write_text(json.dumps(r, indent=2) + '\n')
    from vllm._custom_ops import reshape_and_cache_flash as native
    from tileir_vllm_kv_scatter_v1 import reshape_and_cache_flash as tile
    from norm_runtime_contract_capture_v1.observer import identity
    r['gpu_executed'] = True
    r['native'] = identity(native)

    def load(item):
        return torch.load(item['path'], map_location='cpu', weights_only=True)

    def restore(item):
        cpu = load(item)
        d = item['metadata']
        base = torch.full((d['storage_nbytes'] // cpu.element_size(),), -7, dtype=cpu.dtype, device='cuda')
        v = torch.as_strided(base, d['shape'], d['stride'], d['storage_offset'])
        v.copy_(cpu)
        return (base, v)
    for s in samples:
        assert r['native']['source_sha256'] == s['helper']['source_sha256']
        desc = s['cache_descriptors']
        kd, vd = (desc['key'], desc['value'])
        assert kd['storage_ptr'] == vd['storage_ptr']
        bs = kd['shape'][1]
        slots = torch.tensor(s['slots'], dtype=torch.int64)
        blocks = slots // bs
        offs = slots % bs
        nb = int(blocks.max()) + 1
        shape = [nb] + kd['shape'][1:]
        extent = max((d['storage_offset'] + sum(((n - 1) * st for n, st in zip(shape, d['stride']))) + 1 for d in (kd, vd)))
        assert extent * 2 <= 384 * 1024 * 1024
        initial = torch.full((extent,), -37, dtype=torch.bfloat16)

        def views(base):
            return [torch.as_strided(base, shape, d['stride'], d['storage_offset']) for d in (kd, vd)]
        ik, iv = views(initial)
        for name, v in [('key_cache', ik), ('value_cache', iv)]:
            v[blocks, offs] = load(s['pre_slots'][name])
        expected = initial.clone()
        ek, ev = views(expected)
        for name, v in [('key_cache', ek), ('value_cache', ev)]:
            v[blocks, offs] = load(s['post_slots'][name])
        inp = {k: restore(item) for k, item in s['pre_inputs'].items()}
        before = {k: base.cpu() for k, (base, v) in inp.items()}
        args = {k: v for k, (base, v) in inp.items()}
        input_finite = {k: bool(torch.isfinite(args[k]).all()) for k in ('key', 'value')}
        assert all(input_finite.values()), 'Nonfinite actual K/V input HOLD'
        assert torch.equal(load(s['post_slots']['key_cache']).view(torch.uint8), load(s['pre_inputs']['key']).view(torch.uint8))
        assert torch.equal(load(s['post_slots']['value_cache']).view(torch.uint8), load(s['pre_inputs']['value']).view(torch.uint8))
        previous = {}
        for arm, fn in [('native', native), ('tile', tile)]:
            for repeat in range(2):
                backing = initial.to('cuda')
                kc, vc = views(backing)
                stream_before = torch.cuda.current_stream(backing.device).cuda_stream
                ret = fn(args['key'], args['value'], kc, vc, args['slot_mapping'], s['kv_cache_dtype'], args['k_scale'], args['v_scale'])
                torch.cuda.synchronize()
                actual = backing.cpu()
                ak, av = views(actual)
                selected_metrics = {k: compare(load(s['post_slots'][name]), v[blocks, offs], torch) for k, name, v in [('key', 'key_cache', ak), ('value', 'value_cache', av)]}
                raw = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                preserved = all((torch.equal(base.cpu().view(torch.uint8), before[k].view(torch.uint8)) for k, (base, v) in inp.items()))
                digest = hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()
                repeat_equal = repeat == 0 or previous[arm] == digest
                row = dict(same_stream=torch.cuda.current_stream(backing.device).cuda_stream == stream_before, input_finite=input_finite, selected_metrics=selected_metrics, phase=s['phase'], arm=arm, repeat=repeat, return_none=ret is None, full_owned_storage_raw_equal=raw, input_full_owned_storage_preserved=preserved, repeat_raw_equal=repeat_equal, original_capacity=kd['shape'][0], replay_capacity=nb, owned_storage_bytes=extent * 2, cache_stride=kd['stride'], cache_offsets=[kd['storage_offset'], vd['storage_offset']], shared_backing=kc.untyped_storage().data_ptr() == vc.untyped_storage().data_ptr(), raw_sha256=digest)
                row['passed'] = row['same_stream'] and ret is None and raw and preserved and repeat_equal and row['shared_backing'] and all(input_finite.values()) and selected_gate(selected_metrics)
                r['records'].append(row)
                save()
                if not row['passed']:
                    torch.save(actual, a.output / (s['phase'] + '_' + arm + str(repeat) + '_actual.pt'))
                    r['raw_difference_hold'] = True
                    save()
                    raise AssertionError('Raw replay HOLD')
                previous[arm] = digest
                del backing, kc, vc, actual, ak, av
        del initial, expected, ik, iv, ek, ev, inp, args, before
    r['complete'] = True
    save()
    return r

def run_gpu(torch, output):
    for p, h in dependencies().items():
        assert sha(p) == h, 'Missing/changed external fixture'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=False)
    try:
        result['attention'] = _attention(torch, output / 'attention')
        result['kv'] = _kv(torch, output / 'kv')
        torch.cuda.synchronize()
        result['complete'] = True
        assert audit(result)
        return result
    finally:
        (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
