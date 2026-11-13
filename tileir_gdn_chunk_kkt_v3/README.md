V3 note: V3-CHANGE.md supersedes only MMA K granularity; all numerical/status limitations apply.

V2 note: see V2-CHANGE.md. Historical v1 bitcast proposal below is superseded by explicit supported RZ astype; all other scope limits remain.

# KKT stage: CPU candidate, not numerically qualified

Public vLLM/FLA sources only. SPDX Apache-2.0; original FLA MIT copyright (c) 2023-2025 Songlin Yang, Yu Zhang. Both licenses retained. No proprietary analysis. No GPU/query/compile, numerical result, adoption or speed claim.

Actual saved core inputs: k BF16 [1,1696,16,128], beta FP32 [1,1696,32]. KKT g input is the *stage1 cumsum output*, NOT raw saved core g. Stage1 raw agreement must first be established, or native cumsum must feed both arms to isolate this stage. Output expected FP32 [1,1696,32,64], contiguous, fresh allocation (13,893,632 bytes). Each sequence has53 rows, tile64, 32 heads; head h selects key head floor(h/2).

Source computes scaled_key[i,d] = FP32(k[i,d]) * beta[i] before dot against FP32(k[j,d]). On CUDA `_CAST_DOT_TO_K_DTYPE=False`. All27 public cached TTIR variants use inputPrecision=tf32; JSON default matches. BK32/64/128, warps2/4/8, stages2/3/4 are candidate configs. Cached variants do not identify selected winner or prove a particular launch. Candidate API therefore requires explicit BK and does not pick a winner. Ordered BK blocks pass FP32 accumulator through MMA.

Public generated PTX observed m16n8k8 f32.tf32.tf32.f32 MMA with full FP32 register operands and no explicit cvt.tf32. Candidate uses bitcast-to-tfloat32 rather than silently inserting RN conversion. This is an operand-representation hypothesis, not a claim about undocumented hardware low-bit behavior. It may be rejected or altered by the TileIR compiler; compile and generated code must be inspected after authorized qualification. MMA accumulation ordering can differ even if BK matches.

After dot, multiply by exp(g_i-g_j); no 1/sqrt(K) scale and no LOG2E pre-scaling of cumsum. Public generated exp lowering is FP32 multiply by 0x3fb8aa3b then ex2.approx.f32, so candidate spells that out. Candidate subtraction/multiplication RN, flush_to_zero=False. Actual denormal/compiler behavior remains unqualified. Strict-lower triangular mask removes diagonal/upper and padded columns to +0; store every column0..63 for each actual row0..52 only.

Next isolated replay: native cumsum first, save result; native KKT twice with actual saved k/beta and same cumsum; record public autotuner best_config/cache metadata. Candidate same BK, input descriptors and preserved owners, same output layout; raw/strict/finite/repeat/mask(+0 bytes)/fresh alias checks. Raw mismatch holds; do not relax thresholds. Only successful isolated proof warrants actual24-call shadow. Remaining triangular solve, WY, recurrence and output stages remain unimplemented.
