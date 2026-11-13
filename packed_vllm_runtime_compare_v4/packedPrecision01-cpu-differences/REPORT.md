# CPU-only analysis of unchanged packedPrecision01 HOLD

Original RESULT SHA256 `d554000fb7a3dc980c9504a83facefe4e3d05f7b5df6c06d405ad9373b7a4857`.
Original error: `AssertionError('Public logprob mismatch')`. complete=False remains unchanged.

Completed: engine init, adapter install, three D/F/T semantic requests (32 prompts ×16 tokens each), D/F logger enable/disable, D/F informational comparison. F/T strict numeric gate then failed. Precision pair and adapter restored, engine shutdown recorded.
Executed warmups: 0. Timed requests: 0. Cycles: 0. Static warmup_requests=2/timed_requests=6 describe planned warmups per arm/cycles, not execution. Semantic request durations are instrumented/first-use diagnostics, not performance results.

## Pairwise comparison (left reference for tolerance)

### D_F
- Tokens: 0/512 differ. Key-set cells: 32/512 differ; missing keys left/right 32/32.
- Common logprobs: 2528; numeric differences 2240; rank differences 0; nonfinite values 0.
- Strict tolerance failures on common keys: 1504; max abs 0.125034809113; max error/tolerance 3629.04906.
- Chosen-token numeric differences 480/512, tolerance failures 352/512.
- Worst: {'request': 0, 'step': 15, 'token_id': 20171, 'left': -6.94407320022583, 'right': -7.069108009338379, 'left_rank': 5, 'right_rank': 5, 'abs_error': 0.12503480911254883, 'tolerance': 7.94407320022583e-05, 'tolerance_fail': True, 'chosen_left': False}
- Per generated step numeric/tolerance/key-set mismatch counts: [(0, 0, 0, 0), (1, 160, 0, 0), (2, 32, 0, 0), (3, 160, 96, 0), (4, 160, 160, 0), (5, 160, 64, 0), (6, 160, 160, 0), (7, 160, 160, 0), (8, 128, 128, 32), (9, 160, 160, 0), (10, 160, 0, 0), (11, 160, 32, 0), (12, 160, 160, 0), (13, 160, 160, 0), (14, 160, 160, 0), (15, 160, 64, 0)]
- Per prompt row summary equal across 32 rows: True.

### F_T
- Tokens: 0/512 differ. Key-set cells: 32/512 differ; missing keys left/right 32/32.
- Common logprobs: 2528; numeric differences 2112; rank differences 0; nonfinite values 0.
- Strict tolerance failures on common keys: 1728; max abs 0.125171661377; max error/tolerance 3691.69091.
- Chosen-token numeric differences 448/512, tolerance failures 416/512.
- Worst: {'request': 0, 'step': 3, 'token_id': 318, 'left': -7.391227722167969, 'right': -7.516399383544922, 'left_rank': 4, 'right_rank': 4, 'abs_error': 0.12517166137695312, 'tolerance': 8.391227722167969e-05, 'tolerance_fail': True, 'chosen_left': False}
- Per generated step numeric/tolerance/key-set mismatch counts: [(0, 0, 0, 0), (1, 0, 0, 0), (2, 160, 64, 0), (3, 160, 160, 0), (4, 160, 160, 0), (5, 64, 0, 0), (6, 160, 160, 0), (7, 160, 160, 0), (8, 128, 128, 32), (9, 160, 160, 0), (10, 160, 32, 0), (11, 160, 64, 0), (12, 160, 160, 0), (13, 160, 160, 0), (14, 160, 160, 0), (15, 160, 160, 0)]
- Per prompt row summary equal across 32 rows: True.

### D_T
- Tokens: 0/512 differ. Key-set cells: 0/512 differ; missing keys left/right 0/0.
- Common logprobs: 2560; numeric differences 2400; rank differences 0; nonfinite values 0.
- Strict tolerance failures on common keys: 1664; max abs 0.124989509583; max error/tolerance 3635.0587.
- Chosen-token numeric differences 480/512, tolerance failures 416/512.
- Worst: {'request': 0, 'step': 5, 'token_id': 799, 'left': -10.799583435058594, 'right': -10.674593925476074, 'left_rank': 5, 'right_rank': 5, 'abs_error': 0.12498950958251953, 'tolerance': 0.00011799583435058595, 'tolerance_fail': True, 'chosen_left': False}
- Per generated step numeric/tolerance/key-set mismatch counts: [(0, 0, 0, 0), (1, 160, 0, 0), (2, 160, 64, 0), (3, 160, 160, 0), (4, 160, 32, 0), (5, 160, 64, 0), (6, 160, 160, 0), (7, 160, 160, 0), (8, 160, 160, 0), (9, 160, 160, 0), (10, 160, 32, 0), (11, 160, 32, 0), (12, 160, 160, 0), (13, 160, 160, 0), (14, 160, 160, 0), (15, 160, 160, 0)]
- Per prompt row summary equal across 32 rows: True.

DIFFERENCES.json stores every request/step count, every differing common scalar/rank, exclusive key values, and chosen token IDs. All original files remain unchanged. Python float numeric/FP64 serialized-bit equality does not establish raw BF16 intermediate equality; no full-vocabulary logits or internal layer boundary compared here. F/T mismatch localizes only to runtime output, not a specific operator. No gate relaxation or rerun.
