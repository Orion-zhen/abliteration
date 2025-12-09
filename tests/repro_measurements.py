import torch
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Import our modules (assuming they are in path or relative)
import sys
sys.path.append(".")
from utils.config import ModelConfig, DataConfig, RefusalConfig, AblationConfig
from utils.io import save_measurements, load_measurements
from utils.model import inlayer_results_projection
from utils.math_utils import sparsify_tensor

class TestMeasurements(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.measurements_path = os.path.join(self.output_dir, "measurements.pt")
        
    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_save_load_measurements(self):
        # Create dummy results using proper tensor structure for a dict
        results = {
            "refuse_0": torch.randn(10),
            "harmful_0": torch.randn(10),
            "harmless_0": torch.randn(10)
        }
        layer_scores = {0: 0.5}
        
        # Save
        save_measurements(results, layer_scores, self.measurements_path)
        
        # Load
        loaded_results, loaded_scores = load_measurements(self.measurements_path)
        
        # Verify
        self.assertTrue(torch.allclose(results["refuse_0"], loaded_results["refuse_0"]))
        self.assertEqual(layer_scores[0], loaded_scores[0])

    def test_inlayer_results_projection(self):
        # Setup: Refusal parallel to harmless
        harmless = torch.tensor([1.0, 0.0])
        refusal = torch.tensor([2.0, 0.0]) # Parallel, projection should make it 0
        
        results = {
            "refuse_0": refusal.clone(),
            "harmless_0": harmless.clone()
        }
        
        inlayer_results_projection(results)
        
        # Expect refusal to be 0 (orthogonalized)
        # Proj = (ref . harmless_norm) * harmless_norm = (2 * 1) * [1,0] = [2,0]
        # Ref - Proj = [2,0] - [2,0] = [0,0]
        self.assertTrue(torch.allclose(results["refuse_0"], torch.zeros(2)))

    def test_sparsify_strategies(self):
        vec = torch.tensor([0.1, 0.5, 0.2, 0.9])
        
        # Magnitude (thresh 0.3) -> Keep 0.5, 0.9
        # max is 0.9. fraction 0.3 * 0.9 = 0.27. Keep > 0.27
        res_mag = sparsify_tensor(vec, method="magnitude", threshold=0.3) 
        # 0.1 < 0.27 (0), 0.5 > 0.27 (0.5), 0.2 < 0.27 (0), 0.9 > 0.27 (0.9)
        expected_mag = torch.tensor([0.0, 0.5, 0.0, 0.9])
        self.assertTrue(torch.allclose(res_mag, expected_mag))

        # Percentile (0.5 aka 50%) -> Keep top 50% = 0.5, 0.9
        res_perc = sparsify_tensor(vec, method="percentile", threshold=0.5)
        self.assertTrue(torch.allclose(res_perc, expected_mag))

if __name__ == "__main__":
    unittest.main()
