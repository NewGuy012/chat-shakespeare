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

- **From-scratch GPT**: Modern default scratch architecture (`RMSNorm`, RoPE, QK-Norm, `ReLU²`) with `F.scaled_dot_product_attention`, fused `AdamW`, KV-cached batched sampling, and automatic `gpt2` switching when loading pretrained (revision-pinned) or saved GPT-2 `safetensors` weights (no `transformers` or `pickle`).
- **Hardware-aware training**: [`TrainConfig.auto()`](src/hyperparameters.py) tunes model size, unified `TokenBatcher` placement (on-device `<= 256 MB`, CPU-backed above), precision, and CUDA Graphs (`reduce-overhead`) automatically for CPU (`fp32`), Apple Silicon MPS (`bf16`), and CUDA (`<12GB`, `12–39GB`, and `>=40GB` VRAM tiers).

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

# Single-machine CLI: modern scratch arch, custom BPE tokenizer & hardware defaults
uv run python src/train.py

# Classic GPT-2 architecture or early stopping on val loss plateau
uv run python src/train.py --arch gpt2 --early-stopping 3

# Fine-tune pretrained GPT-2 (automatically selects gpt2 arch & tiktoken BPE)
uv run python src/train.py --init-from gpt2 --lr 3e-4

# Multi-GPU / distributed or custom Accelerate flags
uv run accelerate launch --multi_gpu --mixed_precision bf16 src/train.py
uv run accelerate launch --cpu src/train.py
uv run accelerate launch --mixed_precision bf16 src/train.py --init-from gpt2
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

| Target | Model | Block | Batch × Accum | LR / Beta2 | Precision | Compile |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| `cpu` | 4L / 4H / 128d (~0.92M) | 128 | 16 × 2 | `1.5e-3` / `0.99` | `no` (`fp32`) | `False` |
| `mps` | 6L / 6H / 384d (~11.02M) | 256 | 32 × 2 | `1e-3` / `0.99` | `bf16` | `False` |
| `cuda` (`<12GB`) | 6L / 6H / 384d (~11.02M) | 256 | 32 × 2 | `1e-3` / `0.99` | `bf16` / `fp16` | `True` (`reduce-overhead`) |
| `cuda` (`12–39GB`) | 6L / 6H / 384d (~11.02M) | 256 | 64 × 1 | `1e-3` / `0.99` | `bf16` / `fp16` | `True` (`reduce-overhead`) |
| `cuda` (`>=40GB`) | 6L / 6H / 384d (~11.02M) | 256 | 128 × 1 | `1.4e-3` / `0.95` | `bf16` | `True` (`reduce-overhead`) |

> **Thunder Compute / Cloud GPU Tips:**
> - Persist `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_FX_GRAPH_CACHE=1` on your persistent disk so snapshot restarts skip cold `torch.compile` warmup.
> - [`uv.lock`](uv.lock) resolves PyTorch `2.15` CUDA 13 (`cu13`) wheels on Linux; if a VM image uses an older CUDA 12 driver, point `[tool.uv.sources]` (marked `sys_platform == 'linux'`) at a matching PyTorch CUDA index.

---

BSD 2-Clause [LICENSE](LICENSE) · Moses Yoo · Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and Zero to Hero series.
