# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Logical adaptation of pinned public vLLM helpers. Uncompiled CPU-prepared specialization; not production qualified.
# No registered binding/API. Requires proved typed 1D views of full backing
# and source-equivalent row strides, nonoverlap, index bounds, UVA lifetimes.
import cuda.tile as ct

@ct.kernel
def post_update_kernel(idx, computed, last, counts, sampled, nsampled,
                      nrejected, qstart, tokens, total,
                      SS: ct.Constant[int], TS: ct.Constant[int],
                      CS: ct.Constant[int], HAS_COUNTS: ct.Constant[bool],
                      HAS_QUERY: ct.Constant[bool]):
    req = ct.bid(0)
    state = ct.gather(idx, req)
    if state >= 0:
        old_total = ct.gather(total, state)
        n = ct.gather(nsampled, req)
        if n > 0:
            token = ct.gather(sampled, req * SS + n - 1)
            ct.scatter(last, state, token)
            ct.scatter(total, state, old_total + n)
        # Sequential recurrence preserves repeated-token histogram increments.
        for i in range(n):
            token = ct.gather(sampled, req * SS + i)
            ct.scatter(tokens, state * TS + old_total + i, token)
            if HAS_COUNTS:
                offset = state * CS + token
                count = ct.gather(counts, offset)
                ct.scatter(counts, offset, count + 1)
        qlen = 0
        if HAS_QUERY:
            qlen = ct.gather(qstart, req + 1) - ct.gather(qstart, req)
        delta = qlen - ct.gather(nrejected, req)
        if delta != 0:
            ct.scatter(computed, state, ct.gather(computed, state) + delta)

