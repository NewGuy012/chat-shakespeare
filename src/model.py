from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from hyperparameters import VALID_ARCHS, pad_vocab_size

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
    arch: str = "modern"

    def __post_init__(self) -> None:
        if self.arch not in VALID_ARCHS:
            raise ValueError(
                f"Unsupported arch {self.arch!r}; expected one of {sorted(VALID_ARCHS)}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.arch == "modern" and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError(
                f"arch='modern' requires even head_dim for RoPE, got {self.n_embd // self.n_head}"
            )


class RMSNorm(nn.Module):
    """RMS normalization that casts scale weights to match autocast activation dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, self.normalized_shape, self.weight.to(dtype=x.dtype), self.eps
        )


class ReLU2(nn.Module):
    """Squared ReLU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


def _build_rope_cache(
    block_size: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(block_size, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    c = cos.to(dtype=x.dtype)[None, None, :, :]
    s = sin.to(dtype=x.dtype)[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def _make_norm(config: GPTConfig, dim: int) -> nn.Module:
    if config.arch == "modern":
        return RMSNorm(dim)
    return nn.LayerNorm(dim, bias=config.bias)


class KVCache:
    """Pre-allocated per-layer KV buffers for autoregressive generation."""

    def __init__(self, n_layer: int, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        self.keys: list[torch.Tensor | None] = [None] * n_layer
        self.values: list[torch.Tensor | None] = [None] * n_layer

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos_offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_head, seq_len, head_dim = k.size()
        end_pos = pos_offset + seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"KV cache length {end_pos} exceeds max_seq_len {self.max_seq_len}"
            )
        k_buf = self.keys[layer_idx]
        v_buf = self.values[layer_idx]
        if (
            k_buf is None
            or v_buf is None
            or k_buf.size(0) != bsz
            or k_buf.dtype != k.dtype
            or k_buf.device != k.device
        ):
            shape = (bsz, n_head, self.max_seq_len, head_dim)
            k_buf = torch.empty(shape, dtype=k.dtype, device=k.device)
            v_buf = torch.empty(shape, dtype=v.dtype, device=v.device)
            self.keys[layer_idx] = k_buf
            self.values[layer_idx] = v_buf
        k_buf[:, :, pos_offset:end_pos, :] = k
        v_buf[:, :, pos_offset:end_pos, :] = v
        return k_buf[:, :, :end_pos, :], v_buf[:, :, :end_pos, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )
        head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.q_norm: RMSNorm | None = (
            RMSNorm(head_dim) if config.arch == "modern" else None
        )
        self.k_norm: RMSNorm | None = (
            RMSNorm(head_dim) if config.arch == "modern" else None
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, pos_offset)
            if pos_offset == 0:
                attn_mask = None
                is_causal = T > 1
            elif T == 1:
                attn_mask = None
                is_causal = False
            else:
                attn_mask = torch.ones(
                    T, k.size(2), dtype=torch.bool, device=x.device
                ).tril(diagonal=pos_offset)
                is_causal = False
        else:
            attn_mask = None
            is_causal = True

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act: nn.Module = (
            ReLU2() if config.arch == "modern" else nn.GELU(approximate="tanh")
        )
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.ln_1 = _make_norm(config, config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = _make_norm(config, config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x), rope=rope, pos_offset=pos_offset, kv_cache=kv_cache
        )
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    """Typed container matching GPT-2 ``transformer.*`` state-dict key layout."""

    pos: torch.Tensor | None
    rope_cos: torch.Tensor | None
    rope_sin: torch.Tensor | None

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.arch == "modern":
            self.wpe: nn.Embedding | None = None
            head_dim = config.n_embd // config.n_head
            cos, sin = _build_rope_cache(config.block_size, head_dim)
            self.register_buffer("pos", None, persistent=False)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
            self.register_buffer(
                "pos",
                torch.arange(config.block_size, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            [Block(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f = _make_norm(config, config.n_embd)


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
        if non_embedding and self.transformer.wpe is not None:
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
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        pos_offset: int = 0,
        kv_cache: KVCache | None = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        _, t = idx.size()
        if pos_offset + t > self.config.block_size:
            raise ValueError(
                f"Sequence length {pos_offset + t} exceeds block_size {self.config.block_size}"
            )

        tok_emb = self.transformer.wte(idx)
        rope: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.transformer.wpe is not None:
            pos_buf = self.transformer.pos
            pos = (
                pos_buf[pos_offset : pos_offset + t]
                if pos_buf is not None
                else torch.arange(
                    pos_offset, pos_offset + t, dtype=torch.long, device=idx.device
                )
            )
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)
            cos = self.transformer.rope_cos
            sin = self.transformer.rope_sin
            if cos is not None and sin is not None:
                rope = (
                    cos[pos_offset : pos_offset + t],
                    sin[pos_offset : pos_offset + t],
                )

        for block in self.transformer.h:
            x = block(x, rope=rope, pos_offset=pos_offset, kv_cache=kv_cache)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return (logits if return_logits else None), loss
        return self.lm_head(x[:, [-1], :]), None

    def crop_block_size(self, block_size: int) -> None:
        """Shrink block_size (e.g., when fine-tuning 1024-ctx GPT-2 with shorter context)."""
        if not 1 <= block_size <= self.config.block_size:
            raise ValueError(
                f"block_size must be in [1, {self.config.block_size}], got {block_size}"
            )
        self.config.block_size = block_size
        if self.transformer.wpe is not None:
            self.transformer.wpe.weight = nn.Parameter(
                self.transformer.wpe.weight[:block_size]
            )
            if self.transformer.pos is not None:
                self.transformer.register_buffer(
                    "pos", self.transformer.pos[:block_size], persistent=False
                )
        if (
            self.transformer.rope_cos is not None
            and self.transformer.rope_sin is not None
        ):
            self.transformer.register_buffer(
                "rope_cos", self.transformer.rope_cos[:block_size], persistent=False
            )
            self.transformer.register_buffer(
                "rope_sin", self.transformer.rope_sin[:block_size], persistent=False
            )

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        dropout: float = 0.0,
        weights_path: str | Path | None = None,
        pad_vocab_to: int | None = 64,
    ) -> GPT:
        """Load OpenAI GPT-2 safetensors weights from Hugging Face Hub or a local file."""
        if model_type not in _PRETRAINED_CONFIGS:
            raise ValueError(
                f"Unknown pretrained model {model_type!r}; choose from {sorted(_PRETRAINED_CONFIGS)}"
            )

        raw_vocab_size = 50257
        vocab_size = (
            pad_vocab_size(raw_vocab_size, pad_vocab_to)
            if pad_vocab_to
            else raw_vocab_size
        )
        cfg = GPTConfig(
            **_PRETRAINED_CONFIGS[model_type],
            vocab_size=vocab_size,
            block_size=1024,
            bias=True,
            dropout=dropout,
            arch="gpt2",
        )
        model = cls(cfg)
        if vocab_size > raw_vocab_size:
            with torch.no_grad():
                model.transformer.wte.weight[raw_vocab_size:].zero_()
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
            if (
                target_k == "transformer.wte.weight"
                and src.shape == (raw_vocab_size, cfg.n_embd)
                and vocab_size > raw_vocab_size
            ):
                with torch.no_grad():
                    sd[target_k][:raw_vocab_size].copy_(src)
                loaded.add(target_k)
                continue
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
        valid_vocab_size: int | None = None,
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError(
                "idx must be a non-empty 2D tensor of shape (batch, seq_len)"
            )
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if max_new_tokens == 0:
            return idx

        bsz, prompt_len = idx.size()
        block_size = self.config.block_size
        out = torch.empty(
            (bsz, prompt_len + max_new_tokens), dtype=torch.long, device=idx.device
        )
        out[:, :prompt_len] = idx

        kv_cache = KVCache(self.config.n_layer, block_size)
        valid_limit = (
            min(valid_vocab_size, self.config.vocab_size)
            if valid_vocab_size is not None
            else self.config.vocab_size
        )

        def sample_next(logits: torch.Tensor | None) -> torch.Tensor:
            assert logits is not None
            scaled = logits[:, -1, :].float() / temperature
            if valid_limit < scaled.size(-1):
                scaled[:, valid_limit:] = -float("inf")
            if top_k is not None:
                k = min(top_k, valid_limit)
                if k > 0:
                    v, _ = torch.topk(scaled, k)
                    scaled[scaled < v[:, [-1]]] = -float("inf")
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        ctx_start = max(0, prompt_len - block_size)
        ctx = out[:, ctx_start:prompt_len]
        logits, _ = self(ctx, pos_offset=0, kv_cache=kv_cache)
        cache_len = ctx.size(1)
        out[:, prompt_len] = sample_next(logits).squeeze(-1)

        slide_keep = max(block_size // 2, 1)
        for step in range(1, max_new_tokens):
            cur_end = prompt_len + step
            if cache_len < block_size:
                tok_in = out[:, cur_end - 1 : cur_end]
                logits, _ = self(tok_in, pos_offset=cache_len, kv_cache=kv_cache)
                cache_len += 1
            else:
                ctx = out[:, cur_end - slide_keep : cur_end]
                logits, _ = self(ctx, pos_offset=0, kv_cache=kv_cache)
                cache_len = slide_keep
            out[:, cur_end] = sample_next(logits).squeeze(-1)
        return out
