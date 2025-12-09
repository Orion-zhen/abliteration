import unittest
from unittest.mock import MagicMock, patch, mock_open
import torch
import sys
import os

# Add root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import load_config, ModelConfig
from utils.math_utils import modify_tensor_norm_preserved, sparsify_tensor
from utils.model import welford_gpu_batched_multilayer_float32

class TestAbliterationConfig(unittest.TestCase):
    def setUp(self):
        self.yaml_content = """
        model: "test/model"
        output_dir: "./output/test"
        inference:
          device: "cpu"
        measurements:
            harmful_prompts: "data/harmful.parquet"
        ablation:
            quantile: 0.95
            top_k: 5
            layer_overrides:
                10:
                    scale: 2.0
        """

    def test_load_config(self):
        with patch("builtins.open", mock_open(read_data=self.yaml_content)):
            with patch("os.path.exists", return_value=True):
                config = load_config("fake_config.yaml")
                self.assertEqual(config.model, "test/model")
                self.assertEqual(config.ablation.quantile, 0.95)
                self.assertEqual(config.ablation.layer_overrides[10]["scale"], 2.0)

class TestModifier(unittest.TestCase):
    def test_modify_tensor_norm_preserved(self):
        # Use a weight row that mixes refusal and orthogonal direction
        # W row: [1, 1]. Refusal: [1, 0].
        # Ablating [1, 0] should make the row point more towards [0, 1].
        W = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) 
        refusal = torch.tensor([1.0, 0.0])
        
        # scale=0.5
        new_W = modify_tensor_norm_preserved(W, refusal, scale_factor=0.5)
        
        # Row 0 [1, 1] should change direction. 
        # Row 1 [0, 1] is orthogonal to refusal [1, 0], so checks orthogonality too.
        
        # Check alignment of Row 0
        w0_orig = W[0]
        w0_new = new_W[0]
        
        # Cosine similarity should decrease if they rotated
        cos_sim = torch.dot(w0_orig, w0_new) / (w0_orig.norm() * w0_new.norm())
        self.assertLess(cos_sim, 0.999) # Should be rotated
        
        # Check Row 1 (Orthogonal) - Direction should be preserved
        # cos_sim should be ~1.0
        cos_sim_1 = torch.dot(W[1], new_W[1]) / (W[1].norm() * new_W[1].norm())
        self.assertGreater(cos_sim_1, 0.999) 
        
        # We generally expect W[1] to not change much at all if the projection is exactly 0
        # But depending on Norm Preservation (input vs output), magnitude might shift.
        # This test ensures at least specific directionality logic holds.

class TestSparsify(unittest.TestCase):
    def test_magnitude_sparsify(self):
        vec = torch.tensor([1.0, 0.1, 0.05, 0.8])
        # Max is 1.0. Fraction 0.5 -> threshold 0.5.
        # kept: 1.0, 0.8. Zeroed: 0.1, 0.05
        res = sparsify_tensor(vec, method="magnitude", threshold=0.5)
        self.assertEqual(res[1], 0.0)
        self.assertEqual(res[3], 0.8)

if __name__ == "__main__":
    unittest.main()
