import unittest
import torch
import numpy as np
from utils.math_utils import (
    magnitude_clip,
    remove_orthogonal_projection,
    modify_tensor_simple,
    modify_tensor_norm_preserved,
    sparsify_tensor,
    sparsity_stats,
    magnitude_sparsify,
    percentile_sparsify,
    topk_sparsify,
    soft_threshold_sparsify
)

class TestMathUtils(unittest.TestCase):
    def test_magnitude_clip(self):
        tensor = torch.tensor([-5.0, 0.0, 5.0])
        clipped = magnitude_clip(tensor, max_val=3.0)
        expected = torch.tensor([-3.0, 0.0, 3.0])
        self.assertTrue(torch.allclose(clipped, expected))

    def test_remove_orthogonal_projection(self):
        # v = [1, 0], u = [0, 1] -> v is orthogonal to u, so v should be unchanged
        v = torch.tensor([1.0, 0.0])
        u = torch.tensor([0.0, 1.0])
        result = remove_orthogonal_projection(v, u)
        self.assertTrue(torch.allclose(result, v))

        # v = [1, 1], u = [0, 1] -> projection of v on u is [0, 1], so result should be [1, 0]
        v = torch.tensor([1.0, 1.0])
        u = torch.tensor([0.0, 1.0])
        result = remove_orthogonal_projection(v, u)
        expected = torch.tensor([1.0, 0.0])
        # Note: floating point errors might happen, but strict check for small numbers is usually fine or use allclose
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_modify_tensor_simple(self):
        # W = identity 2x2
        # r = [1, 0]
        # W_new = W - scale * (r @ r.T) @ W
        # r @ r.T = [[1, 0], [0, 0]]
        # (r @ r.T) @ W = [[1, 0], [0, 0]] @ I = [[1, 0], [0, 0]]
        # W_new = I - 1.0 * [[1, 0], [0, 0]] = [[0, 0], [0, 1]]
        
        W = torch.eye(2)
        r = torch.tensor([1.0, 0.0])
        result = modify_tensor_simple(W, r, scale_factor=1.0)
        expected = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_modify_tensor_norm_preserved(self):
        # W with vectors not parallel to r=[1, 0]
        # Row 1: [1, 1] -> norm sqrt(2)
        # Row 2: [0, 1] -> norm 1 (orthogonal to r, so unchanged direction)
        W = torch.tensor([[1.0, 0.0], [1.0, 1.0]]) # [in, out] -> transposed is [[1, 1], [0, 1]]
        # W transposed (as used in calc):
        # Row 0 (matches in=0): [1, 1] (This was col 0 of W)
        # Row 1 (matches in=1): [0, 1] (This was col 1 of W)
        
        # Wait, strictly speaking:
        # W is [in, out].
        # W = [[1, 0], [1, 1]]
        # Col 0: [1, 1]. Norm sqrt(2).
        # Col 1: [0, 1]. Norm 1.
        
        r = torch.tensor([1.0, 0.0]) # Direction to remove (affects first component)
        
        # We want to check if column norms are preserved.
        result = modify_tensor_norm_preserved(W, r, scale_factor=1.0)
        
        # Check if shape is preserved
        self.assertEqual(result.shape, W.shape)
        
        # Check if norms are preserved (per output channel, which is column in [in, out])
        # result is [in, out]
        orig_norms = torch.norm(W, dim=0) # [out]
        new_norms = torch.norm(result, dim=0) # [out]
        self.assertTrue(torch.allclose(orig_norms, new_norms))

    def test_sparsify_tensor_magnitude(self):
        tensor = torch.tensor([1.0, 0.5, 0.01, -0.01, -1.0])
        # threshold = 0.05 * 1.0 = 0.05
        # kept: 1.0, 0.5, -1.0. zeroed: 0.01, -0.01
        result = sparsify_tensor(tensor, method="magnitude", threshold=0.1) 
        # explicitly passing 0.1, threshold = 0.1 * 1.0 = 0.1
        # kept: 1.0, 0.5, -1.0. zeroed: 0.01, -0.01
        expected = torch.tensor([1.0, 0.5, 0.0, 0.0, -1.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_topk_sparsify(self):
        tensor = torch.tensor([0.1, 0.5, 0.2, 0.9])
        # top 2: 0.9, 0.5
        result = topk_sparsify(tensor, k=2)
        expected = torch.tensor([0.0, 0.5, 0.0, 0.9])
        self.assertTrue(torch.allclose(result, expected))

    def test_sparsity_stats(self):
        tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])
        stats = sparsity_stats(tensor)
        self.assertEqual(stats["total_components"], 4)
        self.assertEqual(stats["nonzero_components"], 1)
        self.assertEqual(stats["sparsity"], 0.75)

if __name__ == '__main__':
    unittest.main()
