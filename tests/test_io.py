import unittest
from unittest.mock import MagicMock, patch, mock_open
import torch
import os
import shutil
from pathlib import Path
from utils.io import resolve_model_paths, copy_model_artifacts, save_measurements, load_measurements, load_data

class TestIO(unittest.TestCase):
    @patch("utils.io.cached_file")
    def test_resolve_model_paths_cached_sharded(self, mock_cached):
        # Setup mock for index file
        mock_cached.return_value = "/cache/model.safetensors.index.json"
        
        with patch("builtins.open", mock_open(read_data='{"weight_map": {"layer1": "shard1.safetensors", "layer2": "shard2.safetensors"}}')):
            with patch("json.load", return_value={"weight_map": {"layer1": "shard1.safetensors", "layer2": "shard2.safetensors"}}):
                index_path, model_dir, weight_map, shards = resolve_model_paths("model-id")
        
        self.assertEqual(index_path, "/cache/model.safetensors.index.json")
        self.assertEqual(str(model_dir), "/cache")
        self.assertEqual(len(shards), 2)
        self.assertIn("shard1.safetensors", shards)

    @patch("utils.io.cached_file")
    def test_resolve_model_paths_single_file(self, mock_cached):
        # First call (index) returns None
        # Second call (single file) returns path
        def side_effect(model_id, filename):
            if filename == "model.safetensors.index.json":
                return None
            if filename == "model.safetensors":
                return "/cache/model.safetensors"
            return None
        mock_cached.side_effect = side_effect
        
        index_path, model_dir, weight_map, shards = resolve_model_paths("model-id")
        
        self.assertIsNone(index_path)
        self.assertEqual(str(model_dir), "/cache")
        self.assertEqual(shards, ["model.safetensors"])

    @patch("utils.io.cached_file")
    @patch("shutil.copy")
    @patch("pathlib.Path.exists")
    def test_copy_model_artifacts(self, mock_exists, mock_copy, mock_cached):
        config = MagicMock()
        config.model = "model-id"
        output_path = Path("/output")
        
        # Test copying index
        mock_cached.return_value = "/cache/config.json" # Default for valid files
        mock_exists.return_value = True
        
        copy_model_artifacts(config, output_path, index_path="/original/index.json")
        
        # Check if index was copied
        mock_copy.assert_any_call("/original/index.json", output_path / "model.safetensors.index.json")
        
        # Check if config was copied (since we returned a path for cached_file)
        mock_copy.assert_any_call("/cache/config.json", output_path / "config.json")

    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_measurements(self, mock_makedirs, mock_save):
        results = {"a": 1}
        scores = {"b": 2}
        save_measurements(results, scores, "path/to/save.pt")
        
        mock_makedirs.assert_called_with("path/to", exist_ok=True)
        mock_save.assert_called()
        args, _ = mock_save.call_args
        self.assertEqual(args[0]["results"], results)
        self.assertEqual(args[0]["layer_scores"], scores)

    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_measurements(self, mock_exists, mock_load):
        mock_exists.return_value = True
        mock_load.return_value = {"results": {"a": 1}, "layer_scores": {"b": 2}}
        
        results, scores = load_measurements("path.pt")
        self.assertEqual(results["a"], 1)
        self.assertEqual(scores["b"], 2)

    def test_load_data_txt(self):
        with patch("builtins.open", mock_open(read_data="line1\nline2")):
            data = load_data("file.txt")
            self.assertEqual(data, ["line1\n", "line2"])

    @patch("pandas.read_parquet")
    def test_load_data_parquet(self, mock_pandas):
        df_mock = MagicMock()
        df_mock.get.return_value = MagicMock(tolist=lambda: ["text1", "text2"])
        mock_pandas.return_value = df_mock
        
        data = load_data("file.parquet")
        self.assertEqual(data, ["text1", "text2"])

if __name__ == '__main__':
    unittest.main()
