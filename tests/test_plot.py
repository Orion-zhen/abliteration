import unittest
from unittest.mock import MagicMock, patch
import torch
from utils.plot import analyze_results

class TestPlot(unittest.TestCase):
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    def test_analyze_results(self, mock_subplots, mock_close, mock_save):
        # Mock results for 1 layer
        results = {
            "harmful_0": torch.tensor([1.0, 0.0]),
            "harmless_0": torch.tensor([0.0, 1.0]),
            "refuse_0": torch.tensor([1.0, -1.0]),
            "layers": 1
        }
        
        # Setup mock figures
        mock_fig = MagicMock()
        
        # Manually handling axes indexing
        ax00 = MagicMock()
        ax01 = MagicMock()
        ax10 = MagicMock()
        ax11 = MagicMock()
        
        mock_axes = MagicMock()
        def axes_getitem(key):
            if key == (0, 0): return ax00
            if key == (0, 1): return ax01
            if key == (1, 0): return ax10
            if key == (1, 1): return ax11
            return MagicMock()
        mock_axes.__getitem__.side_effect = axes_getitem
        
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        analyze_results(results, output_dir=".", output_plot_name="test.png")
        
        # Verify plotting calls
        self.assertTrue(ax00.plot.called)
        self.assertTrue(ax01.plot.called)
        self.assertTrue(ax10.plot.called)
        self.assertTrue(ax11.plot.called)
        
        # Verify save
        mock_save.assert_called()

if __name__ == '__main__':
    unittest.main()
