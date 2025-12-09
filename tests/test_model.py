import unittest
from unittest.mock import MagicMock, patch
import torch
from utils.model import welford_gpu_batched_multilayer_float32, compute_refusals, inlayer_results_projection

class TestModel(unittest.TestCase):
    def test_welford_accumulation(self):
        # We simulate a model that returns hidden states
        # The function: welford_gpu_batched_multilayer_float32(prompts, desc, model, tokenizer, layers...)
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda x, **kwargs: {
            'input_ids': torch.tensor([[1, 2]]), 
            'attention_mask': torch.tensor([[1, 1]])
        }
        mock_tokenizer.pad_token = "pad"
        mock_tokenizer.eos_token = "eos"
        
        # Setup mock return from model.generate
        # It returns object with .hidden_states
        # hidden_states is tuple (layers) of tensor [batch, seq, hidden]
        # We care about hidden_states[0] (step 0)
        # Structure: ( (layer0, layer1), ) if it's tuple of tuples?
        # Code says: hidden_states = raw_output.hidden_states[0]
        # then hidden_states[layer_idx]
        
        # Lets simulate 2 layers, hidden size 2
        layer0_out = torch.tensor([[[1.0, 1.0]]]) # Batch 1, Seq 1 (pos -1), Hidden 2
        layer1_out = torch.tensor([[[2.0, 2.0]]])
        
        mock_output = MagicMock()
        mock_output.hidden_states = [(layer0_out, layer1_out)]
        mock_model.generate.return_value = mock_output
        mock_model.device = "cpu"
        
        prompts = ["p1"]
        layers = [0, 1]
        
        means = welford_gpu_batched_multilayer_float32(
            prompts, "desc", mock_model, mock_tokenizer, layers, batch_size=1
        )
        
        self.assertTrue(torch.allclose(means[0], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(means[1], torch.tensor([2.0, 2.0])))

    @patch("utils.model.welford_gpu_batched_multilayer_float32")
    def test_compute_refusals(self, mock_welford):
        # Mock means
        # Layer 0: harmful [1, 0], harmless [0, 1] -> refuse [1, -1] -> norm sqrt(2)
        mock_welford.side_effect = [
            {0: torch.tensor([1.0, 0.0])}, # Harmful
            {0: torch.tensor([0.0, 1.0])}  # Harmless
        ]
        
        mock_model = MagicMock()
        # Ensure hasattr(model, "model") is False to properly trigger fallback to config
        del mock_model.model 
        del mock_model.language_model
        # Also ensure layers doesn't exist on base
        del mock_model.layers
        
        mock_model.config.num_hidden_layers = 1
        mock_tokenizer = MagicMock()
        
        results, scores = compute_refusals(mock_model, mock_tokenizer, ["bad"], ["good"], batch_size=1)
        
        self.assertIn("refuse_0", results)
        refusal = results["refuse_0"]
        expected = torch.tensor([1.0, -1.0])
        self.assertTrue(torch.allclose(refusal, expected))
        
        # Check logic: refuse = harmful - harmless
        
    def test_inlayer_results_projection(self):
        # refuse = [1, 1], harmless = [0, 1]
        # proj of refuse on harmless = [0, 1]
        # new refuse = [1, 0]
        
        results = {
            "refuse_0": torch.tensor([1.0, 1.0]),
            "harmless_0": torch.tensor([0.0, 1.0])
        }
        
        inlayer_results_projection(results)
        
        self.assertTrue(torch.allclose(results["refuse_0"], torch.tensor([1.0, 0.0])))

if __name__ == '__main__':
    unittest.main()
