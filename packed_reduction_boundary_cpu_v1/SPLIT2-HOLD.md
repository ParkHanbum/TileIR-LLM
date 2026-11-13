# Contiguous split2 rejected; original evidence preserved

The first actual ba decode matrix (M32/N64/K2560) stopped both candidates. FP32 partial:768/2048 raw AND strict failures. BF16 partial:864/2048 raw AND strict failures. Both maxabs0.015625, nonfinite0. This is32 repeated X rows, not32 independent numerical cases.

Public D99/math0/compute68 repeats were exact to the stored default native result; the unchanged original TileIR baseline reproduced its saved output. Full logical input and backing/holes preservation passed. Torch flags remained True/True. Both candidates stopped at repeat0 on the first matrix. **Neither second candidate repeat nor any of the remaining four regression matrices ran.** No performance measurement or adoption occurred.

The worker and launcher returned actualexit2 by design; independent recovery confirmed child506180 absent and common lock available. complete=true in RESULT means the planned fail-stop diagnostic completed, NOT numerical qualification. raw_pass=false and qualified_arms=[] are authoritative.

No source/result/receipt has been overwritten or rerun. Result38254fe44e8be5c7a9828d72735c3fed76d6b6cff8a7630335721d105c878213; launchc32c0df1090c73f36a5a16282b6707d4a0071043868ec972e6ceb8a36654bf54; recoverya220e305b8a21c67938ad53bf709d3ef8fbffed6b52913b67a0e77c405b00e0b.

## Next CPU-only distinction
Public CUTLASS LinearCombination::set_k_partition sets beta=1 for subsequent serial K partitions, and its epilogue combines an accumulator with converted prior output before destination conversion. This is different from rounding EVERY independent partial first and reducing them afterward. Source: https://github.com/NVIDIA/cutlass/blob/7127592069c2fe01b041e174ba4345ef9b279671/include/cutlass/epilogue/thread/linear_combination.h (lines195-231 in raw source).

A bounded CPU arithmetic screen will compare serial-output rounding and independent-partial rounding on the saved operands before proposing another GPU experiment. It uses exact integer partial dot→FP32 RN, not simulated tensor-core MMA. Even agreement with a saved GPU output is only endpoint agreement on this case, not an observed internal accumulator or cuBLAS schedule. Public-source arithmetic hypotheses must not be labeled recovered vendor algorithms. No precision policy or tolerance changes.
