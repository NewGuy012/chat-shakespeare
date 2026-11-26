from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

_PRETRAINED_CONFIGS: dict[str, dict[str, int]] = {
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
    "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
}
_CONV1D_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    """Typed container matching GPT-2 ``transformer.*`` state-dict key layout."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying shares token embedding and output projection weights.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=proj_std)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """Shrink block_size (e.g., when fine-tuning 1024-ctx GPT-2 with shorter context)."""
        if not 1 <= block_size <= self.config.block_size:
            raise ValueError(
                f"block_size must be in [1, {self.config.block_size}], got {block_size}"
            )
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        dropout: float = 0.0,
        weights_path: str | Path | None = None,
    ) -> GPT:
        """Load OpenAI GPT-2 safetensors weights from Hugging Face Hub or a local file."""
        if model_type not in _PRETRAINED_CONFIGS:
            raise ValueError(
                f"Unknown pretrained model {model_type!r}; choose from {sorted(_PRETRAINED_CONFIGS)}"
            )

        cfg = GPTConfig(
            **_PRETRAINED_CONFIGS[model_type],
            vocab_size=50257,
            block_size=1024,
            bias=True,
            dropout=dropout,
        )
        model = cls(cfg)
        sd = model.state_dict()

        resolved = (
            Path(weights_path)
            if weights_path is not None
            else Path(
                hf_hub_download(
                    repo_id=f"openai-community/{model_type}",
                    filename="model.safetensors",
                )
            )
        )
        sd_hf = load_file(str(resolved), device="cpu")

        expected = {k for k in sd if k != "lm_head.weight"}
        loaded: set[str] = set()
        for k, tensor in sd_hf.items():
            if k.endswith((".attn.bias", ".attn.masked_bias")) or k == "lm_head.weight":
                continue
            target_k = k if k.startswith("transformer.") else f"transformer.{k}"
            if target_k not in sd:
                raise ValueError(f"Unexpected pretrained weight key: {k!r}")
            src = tensor.t() if any(k.endswith(w) for w in _CONV1D_WEIGHTS) else tensor
            if src.shape != sd[target_k].shape:
                raise ValueError(
                    f"Shape mismatch for {k!r}: {tuple(src.shape)} != {tuple(sd[target_k].shape)}"
                )
            with torch.no_grad():
                sd[target_k].copy_(src)
            loaded.add(target_k)

        if loaded != expected:
            missing = sorted(expected - loaded)
            raise ValueError(f"Missing pretrained weight keys: {missing}")
        return model

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float]
    ) -> torch.optim.AdamW:
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        device_type = params[0].device.type if params else "cpu"
        use_fused = device_type in {"cuda", "mps", "cpu"}
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError(
                "idx must be a non-empty 2D tensor of shape (batch, seq_len)"
            )
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
