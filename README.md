# Chat Shakespeare

A minimal GPT implementation in PyTorch with Hugging Face Accelerate, `safetensors` checkpoints, and a [marimo](https://marimo.io) workflow.

- **From-scratch GPT**: Causal self-attention via `F.scaled_dot_product_attention`, fused `AdamW`, and optional pretrained GPT-2 loading directly from Hugging Face `safetensors` (no `transformers` or `pickle`).
- **Hardware-aware training**: [`TrainConfig.auto()`](src/hyperparameters.py) tunes model size, batching, and precision automatically for CPU (`fp32`), Apple Silicon MPS (`bf16`), and CUDA (`bf16`/`fp16` + `torch.compile`).

## Quickstart

Requires Python `>=3.14` and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NewGuy012/chat-shakespeare.git
cd chat-shakespeare
uv sync
```

Run interactively with marimo (preferred for local exploration) or from the CLI:

```bash
# Preferred locally: interactive notebook (tokenizes, trains, and samples)
uv run marimo edit main.py

# Single-machine CLI: applies TrainConfig.auto() precision & compile defaults
uv run python src/train.py

# Multi-GPU / distributed or custom Accelerate flags
uv run accelerate launch --multi_gpu --mixed_precision bf16 src/train.py
uv run accelerate launch --cpu src/train.py
uv run accelerate launch --mixed_precision bf16 src/train.py --init-from gpt2 --tokenizer gpt2
```

Run tests:

```bash
uv run python -m unittest discover -s tests
```

## Hardware Presets

| Target | Model | Block | Batch × Accum | Precision | Compile |
| :--- | :--- | ---: | ---: | :--- | :--- |
| `cpu` | 4L / 4H / 128d (~0.8M) | 128 | 16 × 2 | `no` (`fp32`) | `False` |
| `mps` | 6L / 6H / 384d (~10.6M) | 256 | 32 × 2 | `bf16` | `False` |
| `cuda` | 6L / 6H / 384d (~10.6M) | 256 | 64 × 1 (32 × 2 if <12GB) | `bf16` / `fp16` | `True` |

---

BSD 2-Clause [LICENSE](LICENSE) · Moses Yoo · Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and Zero to Hero series.
