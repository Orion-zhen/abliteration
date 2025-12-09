import unittest
import os
import tempfile
import yaml
from utils.config import load_config, ModelConfig, InferenceConfig, MeasurementsConfig, AblationConfig

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.test_dir.name, "test_config.yaml")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_config_valid(self):
        config_data = {
            "model": "test-model",
            "output_dir": "./output",
            "inference": {
                "batch_size": 2,
                "max_lengh": 128 # Testing the typo fallback
            },
            "ablation": {
                "method": "biprojection"
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(self.config_path)
        
        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.output_dir, "./output")
        self.assertEqual(config.inference.batch_size, 2)
        self.assertEqual(config.inference.max_length, 128)
        self.assertEqual(config.ablation.method, "biprojection")
        # Defaults
        self.assertEqual(config.measurements.clip, 1.0)

    def test_load_config_missing_required(self):
        # Missing model
        config_data = {"output_dir": "."}
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)
            
        with self.assertRaises(ValueError):
            load_config(self.config_path)

    def test_load_config_missing_output(self):
        # Missing output_dir AND measurements.save_path
        config_data = {"model": "test"}
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)
            
        with self.assertRaises(ValueError):
            load_config(self.config_path)
            
    def test_load_config_typo_handling(self):
        # Verify max_lengh -> max_length
        config_data = {
            "model": "test",
            "output_dir": ".",
            "inference": {"max_lengh": 100}
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)
            
        config = load_config(self.config_path)
        self.assertEqual(config.inference.max_length, 100)

if __name__ == '__main__':
    unittest.main()
