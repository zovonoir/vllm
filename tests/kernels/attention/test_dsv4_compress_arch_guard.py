# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arch-gating contract for the AOT DeepSeek-V4 compressor ops.

Platform-agnostic — runs on any host (including non-ROCm CI). It asserts that
``hip_compressor_supported`` is True exactly when the device is gfx950 (CDNA4)
with the _rocm_C ops built in, and always False otherwise (wrong arch, bf16
state cache off, an unsupported shape, or a non-uint8 cache layout). The model
relies on this gate to choose between the HIP op and the Triton fallback.
"""

import torch

from tests.kernels.attention.dsv4_compress_utils import detect_gfx950
from vllm.models.deepseek_v4.amd.ops.hip_compress_dispatch import (
    SUPPORTED_SHAPES,
    hip_compressor_supported,
)


def test_compressor_gated_on_gfx950():
    on = detect_gfx950()
    if on:
        import vllm._rocm_C  # noqa: F401  (ensure the ops are registered)

    u8 = torch.empty(1, 16, dtype=torch.uint8)

    # Supported iff on gfx950 (all other preconditions satisfied here).
    for head_dim, ratio in SUPPORTED_SHAPES:
        assert hip_compressor_supported(head_dim, ratio, u8, True) == on

    # bf16 state cache off -> never supported (Triton has no bf16 path either).
    assert not hip_compressor_supported(512, 4, u8, False)
    # Unsupported (head_dim, compress_ratio) -> never supported.
    assert not hip_compressor_supported(256, 4, u8, True)
    # Non-uint8 (e.g. FlashInfer full-cache) layout -> never supported.
    bf16 = torch.empty(1, 16, dtype=torch.bfloat16)
    assert not hip_compressor_supported(512, 4, bf16, True)
