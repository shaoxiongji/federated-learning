#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.quantization import (
    quantize_tensor_int4,
    dequantize_tensor_int4,
    compress_state_dict,
    decompress_state_dict,
)
from models.Nets import MLP


class TestQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_tensor_roundtrip_1d(self):
        t = torch.randn(256)
        packed, scales, shape, padded_len, _ = quantize_tensor_int4(t, block_size=64)
        recovered = dequantize_tensor_int4(packed, scales, shape, padded_len, block_size=64)

        self.assertEqual(recovered.shape, t.shape)
        # Check that error is bounded by quantization step (max abs / 7)
        max_err = torch.max(torch.abs(t - recovered)).item()
        scale_est = torch.max(torch.abs(t)).item() / 7.0
        self.assertLess(max_err, scale_est + 1e-4)

    def test_tensor_roundtrip_nd_unaligned(self):
        # 3D tensor with prime number of elements (1377 elements)
        t = torch.randn(9, 17, 9)
        packed, scales, shape, padded_len, _ = quantize_tensor_int4(t, block_size=32)
        recovered = dequantize_tensor_int4(packed, scales, shape, padded_len, block_size=32)

        self.assertEqual(recovered.shape, t.shape)
        cos_sim = torch.cosine_similarity(t.flatten(), recovered.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.98)

    def test_sub_byte_compression_ratio(self):
        t = torch.randn(2048, dtype=torch.float32)
        packed, scales, _, _, _ = quantize_tensor_int4(t, block_size=64)

        fp32_bytes = t.element_size() * t.numel()  # 2048 * 4 = 8192 bytes
        comp_bytes = packed.element_size() * packed.numel() + scales.element_size() * scales.numel()
        # Packed bytes: 1024 bytes, Scales: 32 * 4 = 128 bytes -> 1152 bytes total
        ratio = fp32_bytes / comp_bytes
        self.assertGreater(ratio, 6.5)  # More than 6.5x compression vs float32

    def test_error_feedback_residual_accumulation(self):
        # Over multiple rounds of residual accumulation, error remains bounded
        t = torch.randn(128)
        residual = None
        for _ in range(5):
            packed, scales, shape, padded_len, residual = quantize_tensor_int4(
                t, block_size=32, residual=residual
            )
            rec = dequantize_tensor_int4(packed, scales, shape, padded_len, block_size=32)
            self.assertIsNotNone(residual)

        mean_err = torch.mean(torch.abs(residual)).item()
        self.assertLess(mean_err, 0.15)

    def test_state_dict_compression(self):
        mlp = MLP(dim_in=784, dim_hidden=200, dim_out=10)
        state_dict = mlp.state_dict()

        compressed, residuals = compress_state_dict(state_dict, block_size=64)
        decompressed = decompress_state_dict(compressed)

        for k in state_dict.keys():
            self.assertIn(k, decompressed)
            self.assertEqual(state_dict[k].shape, decompressed[k].shape)
            diff = torch.mean(torch.abs(state_dict[k] - decompressed[k])).item()
            self.assertLess(diff, 0.1)


if __name__ == "__main__":
    unittest.main()
