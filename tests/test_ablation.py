import unittest
from unittest.mock import MagicMock, patch
import torch
from utils.ablation import get_layer_ablation_config, modify_shard_weights, run_sharded_ablation
from utils.config import ModelConfig, AblationConfig

class TestAblation(unittest.TestCase):
    def test_get_layer_ablation_config_default(self):
        config = MagicMock()
        config.ablation.global_scale = 1.0
        config.ablation.layer_overrides = {}
        config.ablation.method = "full"
        config.ablation.sparsify_method = "magnitude"
        
        global_refusal = torch.tensor([1.0, 0.0])
        results = {"harmless_0": torch.tensor([0.0, 1.0])}
        
        scale, refusal, use_norm = get_layer_ablation_config(config, 0, global_refusal, results)
        
        self.assertEqual(scale, 1.0)
        self.assertTrue(use_norm) # "full" implies norm_preserving
        # Check orthogonalization: [1, 0] orth to [0, 1] is [1, 0]
        self.assertTrue(torch.allclose(refusal, torch.tensor([1.0, 0.0])))

    def test_get_layer_ablation_config_override(self):
        config = MagicMock()
        config.ablation.global_scale = 1.0
        config.ablation.layer_overrides = {0: {"scale": 0.5}}
        config.ablation.method = "simple"
        
        global_refusal = torch.tensor([1.0, 0.0])
        results = {}
        
        scale, refusal, use_norm = get_layer_ablation_config(config, 0, global_refusal, results)
        
        self.assertEqual(scale, 0.5)
        self.assertFalse(use_norm)

    @patch("utils.ablation.modify_tensor_norm_preserved")
    def test_modify_shard_weights(self, mock_modify):
        config = MagicMock()
        config.ablation.global_scale = 1.0
        config.ablation.layer_overrides = {}
        config.ablation.method = "full"
        
        global_refusal = torch.tensor([1.0])
        results = {"harmless_0": torch.tensor([0.0])} # dummy
        
        state_dict = {
            "model.layers.0.mlp.down_proj.weight": torch.randn(2, 2)
        }
        
        mock_modify.return_value = torch.zeros(2, 2)
        
        modified = modify_shard_weights(state_dict, config, global_refusal, results)
        
        self.assertTrue(modified)
        self.assertTrue(torch.allclose(state_dict["model.layers.0.mlp.down_proj.weight"], torch.zeros(2, 2)))
        mock_modify.assert_called()

    @patch("utils.ablation.copy_model_artifacts")
    @patch("utils.ablation.save_file")
    @patch("utils.ablation.load_file")
    @patch("utils.ablation.resolve_model_paths")
    def test_run_sharded_ablation(self, mock_resolve, mock_load, mock_save, mock_copy):
        config = MagicMock()
        config.model = "model"
        config.output_dir = "out"
        
        mock_resolve.return_value = (None, "dir", {}, ["shard1"])
        mock_load.return_value = {"model.layers.0.mlp.down_proj.weight": torch.randn(2, 2)}
        
        global_refusal = torch.tensor([1.0])
        results = {"harmless_0": torch.tensor([0.0])}
        
        # We need to make sure modification happens to trigger save
        with patch("utils.ablation.modify_shard_weights", return_value=True):
             run_sharded_ablation(config, global_refusal, results)
             
        mock_save.assert_called()
        mock_copy.assert_called()

if __name__ == '__main__':
    unittest.main()
