#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Block-wise INT4 Quantization with Sub-Byte Bit Packing and Error-Feedback Residuals.
Implements memory-efficient client-server tensor compression for Federated Learning.
"""

import math
from typing import Dict, Tuple, Optional, Any
import torch


def quantize_tensor_int4(
    tensor: torch.Tensor,
    block_size: int = 64,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], int, Optional[torch.Tensor]]:
    """
    Quantizes a floating-point tensor to packed INT4 nibbles with per-block scaling.

    Args:
        tensor: Floating point tensor (float32 or float64).
        block_size: Elements per quantization block.
        residual: Optional residual error from previous round.

    Returns:
        packed_bytes: torch.uint8 tensor where each byte contains two 4-bit nibbles.
        scales: torch.float32 tensor containing the scale factor per block.
        orig_shape: Original tensor shape tuple.
        padded_numel: Number of elements after block padding.
        new_residual: Updated residual tensor for error feedback.
    """
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    orig_device = tensor.device

    flat = tensor.detach().flatten().float()
    if residual is not None:
        flat = flat + residual.to(flat.device).float()

    n = flat.numel()
    if n == 0:
        empty_bytes = torch.empty(0, dtype=torch.uint8, device=orig_device)
        empty_scales = torch.empty(0, dtype=torch.float32, device=orig_device)
        return empty_bytes, empty_scales, orig_shape, 0, None

    # Pad to multiple of (2 * block_size) to ensure even pairs for nibble packing
    align = block_size * 2
    remainder = n % align
    pad_len = (align - remainder) % align
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    padded_numel = padded.numel()
    blocks = padded.view(-1, block_size)

    # Compute per-block max absolute scale
    abs_max = torch.max(torch.abs(blocks), dim=-1, keepdim=True)[0]
    scales = (abs_max / 7.0).clamp(min=1e-8)

    # Quantize to [-8, 7]
    q_blocks = torch.clamp(torch.round(blocks / scales), -8, 7).to(torch.int32)
    q_flat = q_blocks.view(-1)

    # Sub-byte packing: map [-8, 7] to unsigned [0, 15] then pack two nibbles per uint8
    u_vals = (q_flat + 8).to(torch.uint8)
    low_nibbles = u_vals[0::2] & 0x0F
    high_nibbles = (u_vals[1::2] & 0x0F) << 4
    packed_bytes = low_nibbles | high_nibbles

    # Dequantize to calculate exact residual for error feedback
    dq_blocks = q_blocks.float() * scales
    dq_flat = dq_blocks.view(-1)[:n]
    new_residual = (flat[:n] - dq_flat).detach()

    return packed_bytes.to(orig_device), scales.squeeze(-1).to(orig_device), orig_shape, padded_numel, new_residual


def dequantize_tensor_int4(
    packed_bytes: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: Tuple[int, ...],
    padded_numel: int,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantizes packed INT4 nibbles back to floating point tensor.

    Args:
        packed_bytes: Packed uint8 tensor.
        scales: Per-block scales tensor.
        orig_shape: Original shape tuple.
        padded_numel: Number of padded elements.
        block_size: Block size used during quantization.
        dtype: Output tensor dtype.

    Returns:
        Dequantized tensor reshaped to orig_shape.
    """
    if packed_bytes.numel() == 0:
        return torch.empty(orig_shape, dtype=dtype, device=packed_bytes.device)

    packed_flat = packed_bytes.flatten()
    low = (packed_flat & 0x0F).to(torch.int32) - 8
    high = ((packed_flat >> 4) & 0x0F).to(torch.int32) - 8

    # Interleave low and high nibbles
    unpacked = torch.empty(packed_flat.numel() * 2, dtype=torch.int32, device=packed_bytes.device)
    unpacked[0::2] = low
    unpacked[1::2] = high

    unpacked_blocks = unpacked.view(-1, block_size).float()
    scales_view = scales.view(-1, 1)

    dequantized_blocks = unpacked_blocks * scales_view
    flat_out = dequantized_blocks.view(-1)

    num_orig = 1
    for dim in orig_shape:
        num_orig *= dim

    return flat_out[:num_orig].view(orig_shape).to(dtype)


def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    block_size: int = 64,
    residuals: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Compresses an entire PyTorch state dictionary using INT4 block quantization.
    Non-floating point tensors (e.g., integer tracking buffers) are kept unquantized.
    """
    compressed = {}
    new_residuals = {}

    for k, v in state_dict.items():
        if torch.is_floating_point(v) and v.numel() >= block_size:
            res = residuals.get(k) if residuals is not None else None
            packed, scales, shape, padded_len, new_res = quantize_tensor_int4(
                v, block_size=block_size, residual=res
            )
            compressed[k] = {
                "quantized": True,
                "packed": packed,
                "scales": scales,
                "shape": shape,
                "padded_numel": padded_len,
                "block_size": block_size,
                "dtype": v.dtype,
            }
            if new_res is not None:
                new_residuals[k] = new_res
        else:
            compressed[k] = {
                "quantized": False,
                "tensor": v.detach().clone(),
            }

    return compressed, new_residuals


def decompress_state_dict(compressed_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Decompresses an INT4 compressed state dictionary back to standard PyTorch state dict.
    """
    state_dict = {}
    for k, entry in compressed_dict.items():
        if entry.get("quantized", False):
            state_dict[k] = dequantize_tensor_int4(
                packed_bytes=entry["packed"],
                scales=entry["scales"],
                orig_shape=entry["shape"],
                padded_numel=entry["padded_numel"],
                block_size=entry["block_size"],
                dtype=entry["dtype"],
            )
        else:
            state_dict[k] = entry["tensor"]
    return state_dict
