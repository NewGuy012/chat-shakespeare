from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
_VALID_DEVICES = frozenset({"cpu", "mps", "cuda"})
_VALID_PRECISIONS = frozenset({"no", "fp16", "bf16"})


def _detect_device() -> str:
    if os.environ.get("ACCELERATE_USE_CPU", "").lower() == "true":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
            raise ValueError(f"Unsupported device {self.device!r}; expected one of {sorted(_VALID_DEVICES)}")
        if self.mixed_precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"Unsupported mixed_precision {self.mixed_precision!r}; expected one of {sorted(_VALID_PRECISIONS)}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
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
        if self.epoch_iters < 0 or self.warmup_iters < 0 or self.lr_decay_iters < 0 or self.grad_clip < 0:
            raise ValueError("epoch_iters, warmup_iters, lr_decay_iters, and grad_clip must be >= 0")

    @property
    def meta_path(self) -> Path:
        return self.data_path / "meta.json"

    @property
    def train_path(self) -> Path:
        return self.data_path / "train.safetensors"

    @property
    def val_path(self) -> Path:
        return self.data_path / "val.safetensors"

    @property
    def checkpoint_dir(self) -> Path:
        return self.data_path / "checkpoint"

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
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    @classmethod
    def auto(cls, device: str | None = None, **overrides: Any) -> TrainConfig:
        """Return empirically tuned defaults for CPU, Apple MPS, or CUDA GPU."""
        target = _detect_device() if device is None else device
        if target == "cpu":
            preset: dict[str, Any] = {
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
        elif target == "mps":
            mps_bf16 = hasattr(torch.backends.mps, "is_macos_or_newer") and torch.backends.mps.is_macos_or_newer(14, 0)
            preset = {
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
        elif target == "cuda":
            low_vram = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            preset = {
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
        else:
            raise ValueError(f"Unsupported device preset {target!r}; expected one of {sorted(_VALID_DEVICES)}")

        if "max_iters" in overrides and "lr_decay_iters" not in overrides:
            preset["lr_decay_iters"] = overrides["max_iters"]
        preset.update(overrides)
        return cls(**preset)
