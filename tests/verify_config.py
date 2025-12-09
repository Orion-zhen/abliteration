import yaml
import os
import sys
sys.path.append(os.getcwd())
from utils.config import load_config, ModelConfig

def test_config_loading():
    # Helper to create temp config
    def write_config(content, filename="temp_test_config.yaml"):
        with open(filename, "w") as f:
            yaml.dump(content, f)
        return filename

    # Case 1: output_dir only
    c1 = {"model": "gpt2", "output_dir": "out_dir"}
    f1 = write_config(c1)
    cfg1 = load_config(f1)
    assert cfg1.output_dir == "out_dir"
    assert cfg1.measurements.save_path is None
    print("Case 1 passed")

    # Case 2: save_path only
    c2 = {"model": "gpt2", "measurements": {"save_path": "meas_path.pt"}}
    f2 = write_config(c2)
    cfg2 = load_config(f2)
    assert cfg2.output_dir is None
    assert cfg2.measurements.save_path == "meas_path.pt"
    print("Case 2 passed")

    # Case 3: Both
    c3 = {"model": "gpt2", "output_dir": "out_dir", "measurements": {"save_path": "meas_path.pt"}}
    f3 = write_config(c3)
    cfg3 = load_config(f3)
    assert cfg3.output_dir == "out_dir"
    assert cfg3.measurements.save_path == "meas_path.pt"
    print("Case 3 passed")

    # Case 4: Neither (Should fail)
    c4 = {"model": "gpt2"}
    f4 = write_config(c4)
    try:
        load_config(f4)
        print("Case 4 FAILED (Should have raised ValueError)")
    except ValueError as e:
        print(f"Case 4 passed (Caught expected error: {e})")

    # Cleanup
    if os.path.exists("temp_test_config.yaml"):
        os.remove("temp_test_config.yaml")

if __name__ == "__main__":
    test_config_loading()
