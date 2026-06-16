# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production adapter for the fused HIP DeepSeek-V4 compressors (gfx950 / CDNA4).

Bridges the model's ``compress_norm_rope_store_fn(...)`` call site
(``compressor.py``) to the AOT ``torch.ops._rocm_C.dsv4_*`` ops, dispatching by
``(head_dim, compress_ratio)``:

    (512, 4)   -> dsv4_csa_compress
    (512, 128) -> dsv4_hca_compress
    (128, 4)   -> dsv4_indexer_compress   (FP8 or MXFP4 via use_fp4_cache)

These kernels read a **bf16** state cache and add APE in-kernel, emit the legacy
paged UE8M0 / packed layout (``kv_cache`` dtype uint8), and are CDNA4-only — they
are built into ``_rocm_C`` only when gfx950 is among the target archs
(VLLM_ROCM_GFX950). ``hip_compressor_supported`` enforces every precondition.
When a precondition is unmet the model falls back to Triton **only if bf16 state
cache is off** — Triton has no bf16-state-cache path (it raises
NotImplementedError), so when bf16 state cache is on the model raises a clear
root-cause error rather than deferring to a doomed Triton call.
"""

from typing import Any

import torch

SUPPORTED_SHAPES = frozenset({(512, 4), (512, 128), (128, 4)})


def _aot_op(head_dim: int, compress_ratio: int):
    """The torch.ops._rocm_C op for this shape, or None if not registered.

    The op is registered only when _rocm_C was built with VLLM_ROCM_GFX950.
    """
    rocm_C = getattr(torch.ops, "_rocm_C", None)
    if rocm_C is None:
        return None
    if head_dim == 512 and compress_ratio == 4:
        return getattr(rocm_C, "dsv4_csa_compress", None)
    if head_dim == 512 and compress_ratio == 128:
        return getattr(rocm_C, "dsv4_hca_compress", None)
    if head_dim == 128 and compress_ratio == 4:
        return getattr(rocm_C, "dsv4_indexer_compress", None)
    return None


def hip_compressor_supported(
    head_dim: int,
    compress_ratio: int,
    kv_cache: torch.Tensor,
    use_bf16_state_cache: bool,
) -> bool:
    """True iff the fused HIP compressor can serve this configuration.

    Checks (all required): gfx950 (CDNA4), a supported (head_dim, ratio) shape,
    the bf16 state-cache path (so APE is added in-kernel), the legacy paged uint8
    cache layout (the kernels do not emit FlashInfer full-cache rows), and that
    the matching AOT op is actually registered (i.e. _rocm_C was built with
    VLLM_ROCM_GFX950). The caller has already checked the opt-in flag + is_rocm().
    """
    if not use_bf16_state_cache:
        return False
    if (head_dim, compress_ratio) not in SUPPORTED_SHAPES:
        return False
    if kv_cache.dtype != torch.uint8:
        return False
    try:
        from vllm.platforms.rocm import on_gfx950
        if not on_gfx950():
            return False
    except Exception:
        return False
    return _aot_op(head_dim, compress_ratio) is not None


def compress_norm_rope_store_hip(
    *,
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    ape: torch.Tensor,
    use_bf16_state_cache: bool = True,
    **_ignored: Any,
) -> None:
    """Dispatch one fused HIP compressor launch (signature mirrors the model's
    ``compress_norm_rope_store_fn``). Assumes ``hip_compressor_supported`` already
    returned True for this configuration."""
    if num_actual == 0:
        return

    op = _aot_op(head_dim, compress_ratio)
    if op is None:
        raise RuntimeError(
            f"HIP compressor op unavailable for (head_dim={head_dim}, "
            f"compress_ratio={compress_ratio}); was _rocm_C built with "
            f"VLLM_ROCM_GFX950?"
        )

    # Positional args, matching the TORCH_LIBRARY schema in
    # csrc/rocm/torch_bindings.cpp. kv_cache is written in place.
    args = [
        state_cache,
        num_actual,
        ape,  # RAW APE; expanded / added on-device
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # kv_cache_block_size
        scale_dim,
    ]
    if head_dim == 128 and compress_ratio == 4:
        args.append(use_fp4_cache)
    op(*args)
