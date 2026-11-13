"""Closed scalar runtime guard and durable endpoint preservation; no framework import."""
def runtime_guard(engine,policy):
 c=engine.vllm_config
 assert type(engine.engine_core).__name__=='InprocClient'
 assert c.model_config.enforce_eager is True
 assert int(c.compilation_config.mode)==0 and c.compilation_config.cudagraph_mode.name=='NONE'
 assert c.scheduler_config.async_scheduling is True
 policy.select('native')
 return dict(enforce_eager=True,compilation_mode=0,cudagraph_mode='NONE',async_scheduling=True,precision=dict(policy.values()))
def record_endpoints(result,tokens,public,save,exact_tokens,exact_public,ft,fp):
 result['output_token_ids']=tokens;result['public_logprobs']=public;save()
 result['tokens_gate']=exact_tokens(tokens,ft);result['public_gate']=exact_public(public,fp)
