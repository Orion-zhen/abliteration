# Abliteration

Make abliterated models using transformers, easy and fast.

The code has been tested on Llama-3.2, Qwen2.5-Coder, Ministral-8b.

Note: abliteration is not uncensorship. Though abliterated, it doesn't necessarily mean the model is completely uncensored, it simply will not explicitly refuse you.

## Usage

**Clone the repositoty**:

```shell
git clone https://github.com/Orion-zhen/abliteration.git
cd abliteration
```

**Install dependencies**:

```shell
pip install -r requirements.txt
```

**Make your abliterations**:

```shell
python abliteration.py -m <path_to_your_model> -o <output_dir>
```

Now your model will be abliterated and saved to `<output_dir>`. Once it finishes, you can immediately chat with your abliterated model in the terminal. For Chinese models, you can use `--deccp` to abliterate it from certain topics.

```python
lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
  lm_model.layers[layer_idx].self_attn.o_proj.weight.data,
  refusal_dir,
  scale_factor,
)
lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
  lm_model.layers[layer_idx].mlp.down_proj.weight.data,
  refusal_dir,
  scale_factor,
)
```

> Qwen series models are so stubborn that you might need to adjust parameters to make a good abliteration.

Available targets can be found in [transformers model architectures](https://github.com/huggingface/transformers/tree/main/src/transformers/models) and [mergekit model architectures](https://github.com/arcee-ai/mergekit/tree/main/mergekit/_data/architectures).

**Full arguments**:

```shell
usage: abliterate.py [-h] --model MODEL [--device {auto,cuda,cpu}] --output OUTPUT [--skip-begin SKIP_BEGIN] [--skip-end SKIP_END] [--layer-fraction LAYER_FRACTION]
                     [--scale-factor SCALE_FACTOR] [--flash-attn] [--deccp] [--load-in-4bit | --load-in-8bit]

Make abliterated models

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Your model directory or huggingface model ID
  --device {auto,cuda,cpu}, -d {auto,cuda,cpu}
                        Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU
  --output OUTPUT, -o OUTPUT
                        Output directory
  --skip-begin SKIP_BEGIN
                        Number of layers to skip at the beginning. Defaults to 1 to avoid messing with the first layer
  --skip-end SKIP_END   Number of layers to skip at the end
  --layer-fraction LAYER_FRACTION
                        Fraction of layers to use for refusal_dir calculation
  --scale-factor SCALE_FACTOR
                        Scale factor for ablation. Use a negative scale-factor to encourage refusal
  --flash-attn          Use flash attention 2
  --deccp               For Chinese models, in specific topics
  --load-in-4bit        Load model in 4-bit precision using bitsandbytes
  --load-in-8bit        Load model in 8-bit precision using bitsandbytes
```

## Credits

- [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [AUGMXNT/deccp](https://github.com/AUGMXNT/deccp)
- [huihui-ai](https://huggingface.co/huihui-ai)
