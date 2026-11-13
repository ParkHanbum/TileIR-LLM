# SPDX-License-Identifier: Apache-2.0
# Lazy imports; no framework imported by CPU preparation.
def validate(out, index):
    if type(index) is not int or not 0 <= index < 32:
        raise ValueError("only B32 slots 0..31")
    if (tuple(out.shape) != (32,) or tuple(out.stride()) != (1,)
        or str(out.dtype) != "torch.int32" or str(out.device) != "cuda:0"
        or out.storage_offset() != 0 or out.requires_grad):
        raise ValueError("expected owner int32 cuda:0 contiguous offset0 vector32")

def fill_request_slot(out, index):
    validate(out, index)
    import torch
    import cuda.tile as ct
    from runner_scalar32_qualification_v3.kernel import fill_one
    ct.launch(torch.cuda.current_stream(out.device), (1,), fill_one, (out, index))
    # Original fill return is discarded by add_request.
    return None
