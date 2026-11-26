<h1 align="center">
  <img src="docs/logo.png" alt="Chat Shakespeare logo" width="44" valign="middle" />&nbsp; Chat Shakespeare
</h1>

<p align="center">
  A minimal GPT implementation in PyTorch with Hugging Face Accelerate, <code>safetensors</code> checkpoints, and a <a href="https://marimo.io">marimo</a> workflow.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-%3E%3D3.14-3776AB?logo=python&logoColor=white" alt="Python >=3.14" />
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D2.15-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch >=2.15" />
  <img src="https://img.shields.io/badge/Accelerate-%3E%3D1.16-FFD21E?logo=huggingface&logoColor=000" alt="Accelerate >=1.16" />
  <img src="https://img.shields.io/badge/safetensors-%3E%3D0.8-0EA5E9" alt="safetensors >=0.8" />
  <img src="https://img.shields.io/badge/marimo-%3E%3D0.26-1C7D78" alt="marimo >=0.26" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--2--Clause-2563EB" alt="BSD-2-Clause License" /></a>
</p>

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

## Screenshots

### Interactive marimo notebook (`main.py`)
![marimo notebook training and sample output](docs/marimo-notebook.png)

### CLI training & sampling on Apple Silicon (`mps`, `bf16`)
![CLI training and sample output](docs/cli-sample.png)

## Hardware Presets

| Target | Model | Block | Batch × Accum | Precision | Compile |
| :--- | :--- | ---: | ---: | :--- | :--- |
| `cpu` | 4L / 4H / 128d (~0.8M) | 128 | 16 × 2 | `no` (`fp32`) | `False` |
| `mps` | 6L / 6H / 384d (~10.6M) | 256 | 32 × 2 | `bf16` | `False` |
| `cuda` | 6L / 6H / 384d (~10.6M) | 256 | 64 × 1 (32 × 2 if <12GB) | `bf16` / `fp16` | `True` |

---

BSD 2-Clause [LICENSE](LICENSE) · Moses Yoo · Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and Zero to Hero series.
