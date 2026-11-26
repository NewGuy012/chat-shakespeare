from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError, safe_open

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
_VALID_DEVICES = frozenset({"cpu", "mps", "cuda"})
_VALID_PRECISIONS = frozenset({"no", "fp16", "bf16"})
_VALID_ARCHS = frozenset({"modern", "gpt2"})
_DEFAULT_ARCH = "modern"
_VALID_COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune"})
_VALID_TOKENIZERS = frozenset({"bpe", "char", "gpt2"})
_DEFAULT_TOKENIZER = "bpe"
_BPE_MIN_VOCAB_SIZE = 257
_DEFAULT_BPE_VOCAB_SIZE = 1024
_GPT2_MIN_VOCAB_SIZE = 50257
_DEFAULT_GPT2_VOCAB_SIZE = 50304
_PRETRAINED_MODELS = frozenset({"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"})
_PRETRAINED_BLOCK_SIZE = 1024


def pad_vocab_size(vocab_size: int, multiple: int = 64) -> int:
    """Round vocab_size up to a clean multiple for Tensor Core / SIMD alignment."""
    if multiple <= 1:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


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


def _read_safetensors_keys(weights_file: Path) -> set[str] | None:
    if not weights_file.is_file():
        return None
    try:
        with safe_open(str(weights_file), framework="pt", device="cpu") as f:
            return set(f.keys())
    except OSError, SafetensorError:
        return None


def _infer_arch(init_from: str) -> str:
    if init_from in _PRETRAINED_MODELS:
        return "gpt2"
    if init_from == "scratch":
        return _DEFAULT_ARCH
    keys = _read_safetensors_keys(Path(init_from) / "model.safetensors")
    if keys is not None:
        return (
            "gpt2"
            if ("transformer.wpe.weight" in keys or "wpe.weight" in keys)
            else "modern"
        )
    ckpt_cfg = _read_checkpoint_config(init_from)
    if ckpt_cfg is not None:
        arch = ckpt_cfg.get("arch")
        if isinstance(arch, str) and arch in _VALID_ARCHS:
            return arch
        return "gpt2"
    return _DEFAULT_ARCH


def _infer_tokenizer(init_from: str) -> str:
    if init_from in _PRETRAINED_MODELS:
        return "gpt2"
    ckpt_cfg = _read_checkpoint_config(init_from)
    if ckpt_cfg is not None:
        tok = ckpt_cfg.get("tokenizer")
        if isinstance(tok, str) and tok in _VALID_TOKENIZERS:
            return tok
        vocab_size = ckpt_cfg.get("vocab_size")
        if (
            ckpt_cfg.get("init_from") == "scratch"
            and isinstance(vocab_size, int)
            and not isinstance(vocab_size, bool)
            and vocab_size < _BPE_MIN_VOCAB_SIZE
        ):
            return "char"
    return _DEFAULT_TOKENIZER


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
            "compile_mode": "default",
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
            "compile_mode": "default",
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
        vram_gb = 0.0
        bf16 = False
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
            bf16 = torch.cuda.is_bf16_supported()
        if 0.0 < vram_gb < 12.0:
            batch_size, grad_accum = 32, 2
            lr, min_lr, beta2, max_iters = 1e-3, 1e-4, 0.99, 5000
        elif vram_gb >= 40.0:
            batch_size, grad_accum = 128, 1
            lr, min_lr, beta2, max_iters = 1.4e-3, 1.4e-4, 0.95, 2500
        else:
            batch_size, grad_accum = 64, 1
            lr, min_lr, beta2, max_iters = 1e-3, 1e-4, 0.99, 5000
        return {
            "device": "cuda",
            "mixed_precision": "bf16" if bf16 else "fp16",
            "compile": True,
            "compile_mode": "reduce-overhead",
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "block_size": 256,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "dropout": 0.2,
            "learning_rate": lr,
            "min_lr": min_lr,
            "beta2": beta2,
            "max_iters": max_iters,
            "warmup_iters": 100,
            "lr_decay_iters": max_iters,
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
    vocab_size: int = _DEFAULT_BPE_VOCAB_SIZE
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    arch: str = _DEFAULT_ARCH
    init_from: str = "scratch"
    tokenizer: str = _DEFAULT_TOKENIZER
    bpe_vocab_size: int = _DEFAULT_BPE_VOCAB_SIZE
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
    compile_mode: str = "default"
    early_stopping_patience: int = 0
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
        if self.arch not in _VALID_ARCHS:
            raise ValueError(
                f"Unsupported arch {self.arch!r}; expected one of {sorted(_VALID_ARCHS)}"
            )
        if self.compile_mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"Unsupported compile_mode {self.compile_mode!r}; expected one of {sorted(_VALID_COMPILE_MODES)}"
            )
        if self.tokenizer not in _VALID_TOKENIZERS:
            raise ValueError(
                f"Unsupported tokenizer {self.tokenizer!r}; expected one of {sorted(_VALID_TOKENIZERS)}"
            )
        if self.init_from in _PRETRAINED_MODELS:
            self.arch = "gpt2"
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if (
            self.arch == "modern"
            and self.init_from == "scratch"
            and (self.n_embd // self.n_head) % 2 != 0
        ):
            raise ValueError(
                f"arch='modern' requires even head_dim for RoPE, got {self.n_embd // self.n_head}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        positive_ints = (
            "batch_size",
            "block_size",
            "vocab_size",
            "bpe_vocab_size",
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
        if self.tokenizer == "bpe" and self.bpe_vocab_size < _BPE_MIN_VOCAB_SIZE:
            raise ValueError(
                f"bpe_vocab_size must be >= {_BPE_MIN_VOCAB_SIZE} "
                f"(256 byte tokens + <|endoftext|>), got {self.bpe_vocab_size}"
            )
        if (
            self.epoch_iters < 0
            or self.warmup_iters < 0
            or self.lr_decay_iters < 0
            or self.grad_clip < 0
            or self.early_stopping_patience < 0
        ):
            raise ValueError(
                "epoch_iters, warmup_iters, lr_decay_iters, grad_clip, "
                "and early_stopping_patience must be >= 0"
            )

    @property
    def token_dir(self) -> Path:
        name = (
            f"bpe-{self.bpe_vocab_size}" if self.tokenizer == "bpe" else self.tokenizer
        )
        return self.data_path / "tokens" / name

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
        if self.early_stopping_patience > 0:
            payload["early_stopping_patience"] = int(self.early_stopping_patience)
        if self.init_from == "scratch":
            payload.update(
                {
                    "arch": self.arch,
                    "n_layer": int(self.n_layer),
                    "n_head": int(self.n_head),
                    "n_embd": int(self.n_embd),
                    "bias": bool(self.bias),
                }
            )
            if self.tokenizer == "bpe":
                payload["bpe_vocab_size"] = int(self.bpe_vocab_size)
            elif self.tokenizer == "gpt2":
                payload["vocab_size"] = pad_vocab_size(
                    int(self.vocab_size)
                    if int(self.vocab_size) >= _GPT2_MIN_VOCAB_SIZE
                    else _DEFAULT_GPT2_VOCAB_SIZE
                )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @property
    def checkpoint_name(self) -> str:
        block_size = self._effective_block_size()
        tok_label = (
            f"bpe{self.bpe_vocab_size}" if self.tokenizer == "bpe" else self.tokenizer
        )
        if self.init_from == "scratch":
            prefix = f"scratch-{self.arch}-{tok_label}-{self.n_layer}l{self.n_embd}d-b{block_size}"
        elif self.init_from in _PRETRAINED_MODELS:
            prefix = f"{self.init_from}-{tok_label}-b{block_size}"
        else:
            prefix = (
                f"{_slug_part(Path(self.init_from).name)}-{tok_label}-b{block_size}"
            )
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
            "arch": self.arch,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "arch" not in filtered:
            filtered["arch"] = "gpt2"
        if "tokenizer" not in filtered:
            init_from = filtered.get("init_from")
            vocab_size = filtered.get("vocab_size")
            if init_from in _PRETRAINED_MODELS:
                filtered["tokenizer"] = "gpt2"
            elif (
                init_from == "scratch"
                and isinstance(vocab_size, int)
                and not isinstance(vocab_size, bool)
                and vocab_size < _BPE_MIN_VOCAB_SIZE
            ):
                filtered["tokenizer"] = "char"
        if (
            filtered.get("tokenizer") == "bpe"
            and "bpe_vocab_size" not in filtered
            and isinstance(filtered.get("vocab_size"), int)
            and not isinstance(filtered.get("vocab_size"), bool)
            and filtered["vocab_size"] >= _BPE_MIN_VOCAB_SIZE
        ):
            filtered["bpe_vocab_size"] = filtered["vocab_size"]
        return cls(**filtered)

    @classmethod
    def auto(cls, device: str | None = None, **overrides: Any) -> TrainConfig:
        """Return empirically tuned defaults for CPU, Apple MPS, or CUDA GPU."""
        target = _detect_device() if device is None else device
        preset = _device_preset(target)

        if "max_iters" in overrides and "lr_decay_iters" not in overrides:
            preset["lr_decay_iters"] = overrides["max_iters"]

        if "arch" not in overrides and "init_from" in overrides:
            preset["arch"] = _infer_arch(str(overrides["init_from"]))

        target_tokenizer = (
            str(overrides["tokenizer"])
            if "tokenizer" in overrides
            else (
                _infer_tokenizer(str(overrides["init_from"]))
                if "init_from" in overrides
                else _DEFAULT_TOKENIZER
            )
        )
        if "tokenizer" not in overrides and "init_from" in overrides:
            preset["tokenizer"] = target_tokenizer

        if target_tokenizer == "bpe":
            ckpt_cfg = (
                _read_checkpoint_config(str(overrides["init_from"]))
                if "init_from" in overrides
                else None
            )
            if (
                ckpt_cfg is not None
                and ckpt_cfg.get("tokenizer") == "bpe"
                and "bpe_vocab_size" not in overrides
                and "vocab_size" not in overrides
            ):
                inferred_bpe = ckpt_cfg.get("bpe_vocab_size")
                if not isinstance(inferred_bpe, int) or isinstance(inferred_bpe, bool):
                    inferred_bpe = ckpt_cfg.get("vocab_size")
                if (
                    isinstance(inferred_bpe, int)
                    and not isinstance(inferred_bpe, bool)
                    and inferred_bpe >= _BPE_MIN_VOCAB_SIZE
                ):
                    preset["bpe_vocab_size"] = inferred_bpe
                    preset["vocab_size"] = inferred_bpe
            elif "vocab_size" in overrides and "bpe_vocab_size" not in overrides:
                preset["bpe_vocab_size"] = overrides["vocab_size"]
            elif "bpe_vocab_size" in overrides and "vocab_size" not in overrides:
                preset["vocab_size"] = overrides["bpe_vocab_size"]
            elif "vocab_size" not in overrides:
                preset["vocab_size"] = preset.get(
                    "bpe_vocab_size", _DEFAULT_BPE_VOCAB_SIZE
                )
        elif target_tokenizer == "gpt2" and "vocab_size" not in overrides:
            preset["vocab_size"] = _DEFAULT_GPT2_VOCAB_SIZE

        preset.update(overrides)
        return cls(**preset)
