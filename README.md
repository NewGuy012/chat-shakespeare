# Chat Shakespeare

A lean sandbox to explore LLM and PyTorch concepts from scratch.

## Description

- Decoder-only GPT transformer built from the ground up with PyTorch scaled dot-product attention (`F.scaled_dot_product_attention`).
- Trained on `Trelis/tiny-shakespeare` or fine-tuned from pretrained OpenAI GPT-2 `safetensors` weights (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).
- Hardware-aware training, mixed precision, gradient accumulation, and safe checkpointing powered by Hugging Face `accelerate` and `safetensors` (no `pickle` / `torch.load`).

## Getting Started

### Requirements & Dependencies

- Python `>=3.14` and [`uv`](https://docs.astral.sh/uv/)
- `torch`, `accelerate`, `safetensors`, `datasets`, `huggingface-hub`, `tiktoken`, `marimo`

### Setup & Running

```bash
git clone https://github.com/NewGuy012/chat-shakespeare.git
cd chat-shakespeare/
uv sync
```

**Interactive notebook (marimo):**

```bash
.venv/bin/marimo edit main.py
```

**CLI / multi-GPU via Hugging Face Accelerate:**

```bash
# Auto-detects CPU (fp32), Apple Silicon MPS (bf16), or CUDA (bf16/fp16 + compile)
.venv/bin/accelerate launch src/train.py

# Force CPU preset or fine-tune pretrained GPT-2 weights
.venv/bin/accelerate launch --cpu src/train.py
.venv/bin/accelerate launch src/train.py --init-from gpt2 --tokenizer gpt2
```

**Run tests:**

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Hardware Presets (`TrainConfig.auto()`)

| Target | Model | Block | Batch × Grad Accum | Precision | Compile |
| :--- | :--- | ---: | ---: | :--- | :--- |
| `cpu` | 4L / 4H / 128d (~0.8M) | 128 | 16 × 2 | `no` (`fp32`) | `False` |
| `mps` | 6L / 6H / 384d (~10.6M) | 256 | 32 × 2 | `bf16` | `False` |
| `cuda` | 6L / 6H / 384d (~10.6M) | 256 | 64 × 1 (32 × 2 if <12GB) | `bf16` / `fp16` | `True` |

Override any setting directly via `TrainConfig.auto(max_iters=500, init_from="gpt2")`.

## Authors

Moses Yoo, juyoung.m.yoo at gmail dot com.

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the BSD 2-Clause license. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project was inspired by Andrej Karpathy's Zero to Hero YouTube playlist as well as his nanoGPT repo.
