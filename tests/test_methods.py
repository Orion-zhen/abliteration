
import torch
import unittest
import numpy as np
import numpy as np
from unittest.mock import patch
from utils.math_utils import modify_tensor_simple, modify_tensor_norm_preserved

class TestAbliterationMethods(unittest.TestCase):
    def setUp(self):
        # Force CPU
        self.device = 'cpu'
        self.patcher = patch('torch.cuda.is_available', return_value=False)
        self.patcher.start()
        
        self.out_features = 10
        self.in_features = 20
        self.W = torch.randn(self.out_features, self.in_features)
        
        # Create random refusal direction [out_features]
        # In reality, refusal is in hidden size.
        # If W is o_proj [hidden, hidden], then refusal corresponds to rows?
        # modify_tensor_simple: matmul(outer(r,r), W).
        # r: [out], W: [out, in].
        # result: [out, in].
        # So r matches out_features.
        self.refusal_dir = torch.randn(self.out_features)
        self.refusal_dir = torch.nn.functional.normalize(self.refusal_dir, dim=0)

    def tearDown(self):
        self.patcher.stop()

    def test_simple_modification(self):
        """Test modify_tensor_simple"""
        scale = 1.0
        W_mod = modify_tensor_simple(self.W.clone(), self.refusal_dir, scale)
        
        # Check if modified
        self.assertFalse(torch.allclose(self.W, W_mod))
        
        # Check orthogonality: simple implementation removes the component p = r r^T W
        # The rows (columns?) component aligned with r should be removed?
        # Formula: W_new = W - scale * r r^T W.
        # This acts on dim 0.
        # So r^T W_new (projection of columns onto r) should be reduced?
        # matmul(r, W_new) = r^T (I - s r r^T) W = (r^T - s r^T r r^T) W
        # if r is unit, r^T r = 1.
        # = (r^T - s r^T) W = (1-s) r^T W.
        # If scale=1, should be 0.
        
        proj_original = torch.matmul(self.refusal_dir, self.W)
        proj_new = torch.matmul(self.refusal_dir, W_mod)
        
        print(f"\nSimple: Proj Original Norm: {proj_original.norm().item()}")
        print(f"Simple: Proj New Norm: {proj_new.norm().item()}")
        
        self.assertTrue(proj_new.norm().item() < 1e-5, "Projected component should be removed in simple ablation")

        # Check Norm Preservation (Should FAIL)
        # We check row norms? No, column norms logic.
        # But wait, modify_tensor_norm_preserved preserves ROW norms of W (if W is linear)?
        # Let's check logic in norm_preserving.
        pass

    def test_norm_preserving_modification(self):
        """Test modify_tensor_norm_preserved"""
        scale = 1.0
        # modify_tensor_norm_preserved expects Safetensors format [in, out] in comments, but code transposes.
        # Wait, my analysis concluded it handles [out, in] input correctly if the comment "Safetensors format" meant "what safetensors returns" which is [out, in].
        # Let's try passing [out, in] W.
        
        W_mod = modify_tensor_norm_preserved(self.W.clone(), self.refusal_dir, scale)
        
        self.assertFalse(torch.allclose(self.W, W_mod))
        
        # Check Norm Preservation
        # modify_tensor_norm_preserved calculates W_norm = norm(W.T(in, out), dim=1) -> [in].
        # Wait, if W is [out, in], W.T is [in, out].
        # Norm dim 1 of [in, out] is sum of squares of each row (output vector).
        # So it preserves norm of output vectors? i.e. Column norms of original W?
        # Let's check what it preserves.
        
        # If logic is:
        # W_gpu = W.T [in, out]
        # W_norm = W_gpu.norm(dim=1) -> [in]
        # W_new = W_norm * W_dir_new
        # result = W_new.T [out, in]
        
        # Then W_new rows (dim 0, size in) have same norm as W_gpu rows.
        # So W_mod columns (dim ??) have same norm.
        # W_mod.T rows match W.T rows.
        
        # So W_mod column j should have same norm as W column j.
        # Column j of W ([out, in]) is vector W[:, j].
        
        norms_orig = torch.norm(self.W, dim=0) # [in]
        norms_new = torch.norm(W_mod, dim=0)   # [in]
        
        print(f"Norm Preserving: Max Diff in Column Norms: {(norms_orig - norms_new).abs().max().item()}")
        self.assertTrue(torch.allclose(norms_orig, norms_new, atol=1e-5), "Column norms should be preserved")


if __name__ == "__main__":
    unittest.main()
