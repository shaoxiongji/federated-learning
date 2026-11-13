#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, Optional, Any
import torch


def quantize_tensor_int4(
    tensor: torch.Tensor,
    block_size: int = 64,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], Optional[torch.Tensor]]:
    """Quantize a floating-point tensor into packed INT4 nibbles with per-block scales."""
    orig_shape = tensor.shape
    orig_device = tensor.device

    flat = tensor.detach().flatten().float()
    if residual is not None:
        flat = flat + residual.to(flat.device).float()

    n = flat.numel()
    if n == 0:
        empty_bytes = torch.empty(0, dtype=torch.uint8, device=orig_device)
        empty_scales = torch.empty(0, dtype=torch.float32, device=orig_device)
        return empty_bytes, empty_scales, orig_shape, None

    # Pad to a multiple of 2 * block_size so every uint8 holds two 4-bit nibbles.
    align = block_size * 2
    pad_len = (align - (n % align)) % align
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    blocks = padded.view(-1, block_size)
    abs_max = torch.max(torch.abs(blocks), dim=-1, keepdim=True)[0]
    scales = (abs_max / 7.0).clamp(min=1e-8)

    q_blocks = torch.clamp(torch.round(blocks / scales), -8, 7).to(torch.int32)
    q_flat = q_blocks.view(-1)

    u_vals = (q_flat + 8).to(torch.uint8)
    packed_bytes = (u_vals[0::2] & 0x0F) | ((u_vals[1::2] & 0x0F) << 4)

    dq_flat = (q_blocks.float() * scales).view(-1)[:n]
    new_residual = (flat[:n] - dq_flat).detach()

    return packed_bytes.to(orig_device), scales.squeeze(-1).to(orig_device), orig_shape, new_residual


def dequantize_tensor_int4(
    packed_bytes: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: Tuple[int, ...],
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed INT4 nibbles back into a floating-point tensor."""
    if packed_bytes.numel() == 0:
        return torch.empty(orig_shape, dtype=dtype, device=packed_bytes.device)

    packed_flat = packed_bytes.flatten()
    low = (packed_flat & 0x0F).to(torch.int32) - 8
    high = ((packed_flat >> 4) & 0x0F).to(torch.int32) - 8

    unpacked = torch.empty(packed_flat.numel() * 2, dtype=torch.int32, device=packed_bytes.device)
    unpacked[0::2] = low
    unpacked[1::2] = high

    dequantized = unpacked.view(-1, block_size).float() * scales.view(-1, 1)
    num_orig = 1
    for dim in orig_shape:
        num_orig *= dim

    return dequantized.view(-1)[:num_orig].view(orig_shape).to(dtype)


def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    block_size: int = 64,
    residuals: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Compress floating-point tensors in a state dict to packed INT4."""
    compressed = {}
    new_residuals = {}

    for k, v in state_dict.items():
        if torch.is_floating_point(v) and v.numel() >= block_size:
            res = residuals.get(k) if residuals is not None else None
            packed, scales, shape, new_res = quantize_tensor_int4(
                v, block_size=block_size, residual=res
            )
            compressed[k] = {
                "quantized": True,
                "packed": packed,
                "scales": scales,
                "shape": shape,
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
    """Restore a state dict compressed with compress_state_dict."""
    state_dict = {}
    for k, entry in compressed_dict.items():
        if entry.get("quantized", False):
            state_dict[k] = dequantize_tensor_int4(
                packed_bytes=entry["packed"],
                scales=entry["scales"],
                orig_shape=entry["shape"],
                block_size=entry["block_size"],
                dtype=entry["dtype"],
            )
        else:
            state_dict[k] = entry["tensor"]
    return state_dict
