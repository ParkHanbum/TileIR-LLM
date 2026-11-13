"""Fail closed; native capture completion never means TileIR qualification."""
def audit(r):
    try:
        assert r['complete'] and r['engine_shutdown'] and r['math_flags_unchanged'] and not r.get('cleanup_errors')
        assert r['pre_engine_math_flags']==r['initial_math_flags']==[True,True]
        req=r['warmup_records']+r['samples'];assert len(req)==2
        assert all(x['generated_tokens']==512 and x['cross_runtime_tokens_equal'] is True and x['output_token_ids']==req[0]['output_token_ids'] for x in req)
        g=r['public_logprob_gate'];assert g['passed'] and g['exact_numeric'] and g['values']==2560 and g['max_abs']==0
        m,=r['capture_manifest'];assert m['schema']=='vllm-unpacked-five-contract-native-v1' and m['complete'] and m['observers_removed']
        assert len(m['modules'])==65 and len(m['counts'])==65 and all(x==16 for x in m['counts'].values())
        assert len(m['rows'])==1040 and all(x['passed'] and x['original_calls']==1 for x in m['rows'])
        assert len(m['samples'])==8 and {(x['kind'],x['phase']) for x in m['samples']}=={(k,p) for k in ('gdn_out','attn_out','down','head') for p in ('prefill','decode')}
        assert all(x['X_preserved'] and x['returned_native_object'] and x['flags_before']==x['flags_after']==[True,True] and x['public_api']['context']['math_mode']=='CUBLAS_DEFAULT_MATH' and x['public_logger']['disabled_success'] and x['public_logger']['off_verified_no_file_change'] for x in m['samples'])
        assert len(m['weights'])==4 and all(x['full_named_tensor_byte_equal'] and x['logical_sha256']==x['checkpoint_named_tensor_sha256']==x['final_logical_sha256'] and x['weight_files_created']==0 for x in m['weights'].values())
        assert m['artifact_logical_bytes']<=192*1024*1024
        return dict(passed=True,errors=[],tileir_qualified=False)
    except (KeyError,AssertionError,TypeError,ValueError):return dict(passed=False,errors=['native capture HOLD'],tileir_qualified=False)
