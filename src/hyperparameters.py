from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
_VALID_DEVICES = frozenset({"cpu", "mps", "cuda"})
_VALID_PRECISIONS = frozenset({"no", "fp16", "bf16"})
_VALID_TOKENIZERS = frozenset({"char", "gpt2"})
_PRETRAINED_MODELS = frozenset({"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"})
_PRETRAINED_BLOCK_SIZE = 1024


def _read_checkpoint_config(init_from: str) -> dict[str, Any] | None:
    if init_from == "scratch" or init_from in _PRETRAINED_MODELS:
        return None
    cfg_file = Path(init_from) / "config.json"
    if not cfg_file.is_file():
        return None
    try:
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _infer_tokenizer(init_from: str) -> str:
    if init_from in _PRETRAINED_MODELS:
        return "gpt2"
    ckpt_cfg = _read_checkpoint_config(init_from)
    if ckpt_cfg is not None:
        tok = ckpt_cfg.get("tokenizer")
        if isinstance(tok, str) and tok in _VALID_TOKENIZERS:
            return tok
    return "char"


def _base_block_size(init_from: str) -> int | None:
    if init_from in _PRETRAINED_MODELS:
        return _PRETRAINED_BLOCK_SIZE
    ckpt_cfg = _read_checkpoint_config(init_from)
    if ckpt_cfg is not None:
        block_size = ckpt_cfg.get("block_size")
        if isinstance(block_size, int) and not isinstance(block_size, bool):
            return block_size
    return None


def _canonical_init_from(init_from: str) -> str:
    if init_from == "scratch" or init_from in _PRETRAINED_MODELS:
        return init_from
    path = Path(init_from)
    return str(path.resolve()) if path.exists() else init_from


def _slug_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "ckpt"


def _detect_device() -> str:
    if os.environ.get("ACCELERATE_USE_CPU", "").lower() == "true":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_preset(target: str) -> dict[str, Any]:
    """Return baseline TrainConfig field values for a validated target device."""
    if target == "cpu":
        return {
            "device": "cpu",
            "mixed_precision": "no",
            "compile": False,
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "block_size": 128,
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "dropout": 0.1,
            "learning_rate": 1.5e-3,
            "min_lr": 1.5e-4,
            "max_iters": 1000,
            "warmup_iters": 50,
            "lr_decay_iters": 1000,
            "eval_iters": 10,
            "log_iters": 100,
        }
    if target == "mps":
        mps_bf16 = hasattr(
            torch.backends.mps, "is_macos_or_newer"
        ) and torch.backends.mps.is_macos_or_newer(14, 0)
        return {
            "device": "mps",
            "mixed_precision": "bf16" if mps_bf16 else "no",
            "compile": False,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "block_size": 256,
            "batch_size": 32,
            "gradient_accumulation_steps": 2,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "min_lr": 1e-4,
            "max_iters": 3000,
            "warmup_iters": 100,
            "lr_decay_iters": 3000,
            "eval_iters": 20,
            "log_iters": 250,
        }
    if target == "cuda":
        low_vram = (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3
        )
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return {
            "device": "cuda",
            "mixed_precision": "bf16" if bf16 else "fp16",
            "compile": True,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "block_size": 256,
            "batch_size": 32 if low_vram else 64,
            "gradient_accumulation_steps": 2 if low_vram else 1,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "min_lr": 1e-4,
            "max_iters": 5000,
            "warmup_iters": 100,
            "lr_decay_iters": 5000,
            "eval_iters": 20,
            "log_iters": 250,
        }
    raise ValueError(
        f"Unsupported device preset {target!r}; expected one of {sorted(_VALID_DEVICES)}"
    )


@dataclass
class TrainConfig:
    """Hyperparameters, paths, and Accelerate runtime settings for GPT training."""

    root_path: Path = field(default_factory=lambda: _DEFAULT_ROOT)
    data_path: Path = field(default_factory=lambda: _DEFAULT_ROOT / "data")
    batch_size: int = 16
    block_size: int = 128
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    init_from: str = "scratch"
    tokenizer: str = "char"
    weight_decay: float = 0.1
    learning_rate: float = 1.5e-3
    min_lr: float = 1.5e-4
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    max_iters: int = 1000
    epoch_iters: int = 0
    warmup_iters: int = 50
    lr_decay_iters: int = 1000
    device: str = "cpu"
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 2
    compile: bool = False
    eval_iters: int = 10
    log_iters: int = 100
    seed: int | None = 42

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path)
        self.data_path = Path(self.data_path)
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"Unsupported device {self.device!r}; expected one of {sorted(_VALID_DEVICES)}"
            )
        if self.mixed_precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"Unsupported mixed_precision {self.mixed_precision!r}; expected one of {sorted(_VALID_PRECISIONS)}"
            )
        if self.tokenizer not in _VALID_TOKENIZERS:
            raise ValueError(
                f"Unsupported tokenizer {self.tokenizer!r}; expected one of {sorted(_VALID_TOKENIZERS)}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        positive_ints = (
            "batch_size",
            "block_size",
            "vocab_size",
            "n_layer",
            "n_head",
            "n_embd",
            "max_iters",
            "gradient_accumulation_steps",
            "eval_iters",
            "log_iters",
        )
        for name in positive_ints:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if (
            self.epoch_iters < 0
            or self.warmup_iters < 0
            or self.lr_decay_iters < 0
            or self.grad_clip < 0
        ):
            raise ValueError(
                "epoch_iters, warmup_iters, lr_decay_iters, and grad_clip must be >= 0"
            )

    @property
    def token_dir(self) -> Path:
        return self.data_path / "tokens" / self.tokenizer

    @property
    def meta_path(self) -> Path:
        return self.token_dir / "meta.json"

    @property
    def train_path(self) -> Path:
        return self.token_dir / "train.safetensors"

    @property
    def val_path(self) -> Path:
        return self.token_dir / "val.safetensors"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_path / "checkpoints"

    def _effective_block_size(self) -> int:
        base_block = _base_block_size(self.init_from)
        return (
            min(int(self.block_size), base_block)
            if base_block is not None
            else int(self.block_size)
        )

    def fingerprint(self) -> str:
        """Return SHA-256 hex digest of cache-relevant training configuration."""
        payload: dict[str, Any] = {
            "init_from": _canonical_init_from(self.init_from),
            "tokenizer": self.tokenizer,
            "block_size": self._effective_block_size(),
            "dropout": float(self.dropout),
            "batch_size": int(self.batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "weight_decay": float(self.weight_decay),
            "learning_rate": float(self.learning_rate),
            "min_lr": float(self.min_lr),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "grad_clip": float(self.grad_clip),
            "max_iters": int(self.max_iters) if self.epoch_iters == 0 else None,
            "epoch_iters": int(self.epoch_iters),
            "warmup_iters": int(self.warmup_iters),
            "lr_decay_iters": int(self.lr_decay_iters),
            "mixed_precision": self.mixed_precision,
            "seed": None if self.seed is None else int(self.seed),
        }
        if self.init_from == "scratch":
            payload.update(
                {
                    "n_layer": int(self.n_layer),
                    "n_head": int(self.n_head),
                    "n_embd": int(self.n_embd),
                    "bias": bool(self.bias),
                }
            )
            if self.tokenizer == "gpt2":
                payload["vocab_size"] = max(int(self.vocab_size), 50257)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @property
    def checkpoint_name(self) -> str:
        block_size = self._effective_block_size()
        if self.init_from == "scratch":
            prefix = (
                f"scratch-{self.tokenizer}-{self.n_layer}l{self.n_embd}d-b{block_size}"
            )
        elif self.init_from in _PRETRAINED_MODELS:
            prefix = f"{self.init_from}-{self.tokenizer}-b{block_size}"
        else:
            prefix = f"{_slug_part(Path(self.init_from).name)}-{self.tokenizer}-b{block_size}"
        return f"{prefix}-{self.fingerprint()[:12]}"

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoints_dir / self.checkpoint_name

    @property
    def download_path(self) -> Path:
        return self.data_path / "download"

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if (
            "tokenizer" not in filtered
            and filtered.get("init_from") in _PRETRAINED_MODELS
        ):
            filtered["tokenizer"] = "gpt2"
        return cls(**filtered)

    @classmethod
    def auto(cls, device: str | None = None, **overrides: Any) -> TrainConfig:
        """Return empirically tuned defaults for CPU, Apple MPS, or CUDA GPU."""
        target = _detect_device() if device is None else device
        preset = _device_preset(target)

        if "max_iters" in overrides and "lr_decay_iters" not in overrides:
            preset["lr_decay_iters"] = overrides["max_iters"]
        if "tokenizer" not in overrides and "init_from" in overrides:
            preset["tokenizer"] = _infer_tokenizer(str(overrides["init_from"]))
        preset.update(overrides)
        return cls(**preset)
