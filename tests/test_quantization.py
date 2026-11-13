#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import torch

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
        packed, scales, shape, _ = quantize_tensor_int4(t, block_size=64)
        recovered = dequantize_tensor_int4(packed, scales, shape, block_size=64)

        self.assertEqual(recovered.shape, t.shape)
        max_err = torch.max(torch.abs(t - recovered)).item()
        scale_est = torch.max(torch.abs(t)).item() / 7.0
        self.assertLess(max_err, scale_est + 1e-4)

    def test_tensor_roundtrip_nd_unaligned(self):
        t = torch.randn(9, 17, 9)
        packed, scales, shape, _ = quantize_tensor_int4(t, block_size=32)
        recovered = dequantize_tensor_int4(packed, scales, shape, block_size=32)

        self.assertEqual(recovered.shape, t.shape)
        cos_sim = torch.cosine_similarity(t.flatten(), recovered.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.98)

    def test_sub_byte_compression_ratio(self):
        t = torch.randn(2048, dtype=torch.float32)
        packed, scales, _, _ = quantize_tensor_int4(t, block_size=64)

        fp32_bytes = t.element_size() * t.numel()
        comp_bytes = packed.element_size() * packed.numel() + scales.element_size() * scales.numel()
        self.assertGreater(fp32_bytes / comp_bytes, 6.5)

    def test_error_feedback_residual_accumulation(self):
        t = torch.randn(128)
        residual = None
        for _ in range(5):
            packed, scales, shape, residual = quantize_tensor_int4(
                t, block_size=32, residual=residual
            )
            _ = dequantize_tensor_int4(packed, scales, shape, block_size=32)
            self.assertIsNotNone(residual)

        self.assertLess(torch.mean(torch.abs(residual)).item(), 0.15)

    def test_state_dict_compression(self):
        mlp = MLP(dim_in=784, dim_hidden=200, dim_out=10)
        state_dict = mlp.state_dict()

        compressed, _ = compress_state_dict(state_dict, block_size=64)
        decompressed = decompress_state_dict(compressed)

        for k in state_dict:
            self.assertIn(k, decompressed)
            self.assertEqual(state_dict[k].shape, decompressed[k].shape)
            self.assertLess(torch.mean(torch.abs(state_dict[k] - decompressed[k])).item(), 0.1)


if __name__ == "__main__":
    unittest.main()
