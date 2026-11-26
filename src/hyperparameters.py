from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeGuard

import torch
from safetensors import SafetensorError, safe_open

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
VALID_DEVICES = ("cpu", "mps", "cuda")
VALID_PRECISIONS = ("no", "fp16", "bf16")
VALID_ARCHS = ("modern", "gpt2")
VALID_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
VALID_TOKENIZERS = ("bpe", "char", "gpt2")
DEFAULT_ARCH = "modern"
DEFAULT_TOKENIZER = "bpe"
BPE_MIN_VOCAB_SIZE = 257
DEFAULT_BPE_VOCAB_SIZE = 1024
GPT2_VOCAB_SIZE = 50257
DEFAULT_GPT2_VOCAB_SIZE = 50304
PRETRAINED_BLOCK_SIZE = 1024
VOCAB_ALIGN = 64
PRETRAINED_CONFIGS: dict[str, dict[str, int]] = {
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
    "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
}
PRETRAINED_REVISIONS: dict[str, str] = {
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "gpt2-medium": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
    "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
    "gpt2-xl": "15ea56dee5df4983c59b2538573817e1667135e2",
}
_GPU_PRESET: dict[str, Any] = {
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "block_size": 256,
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


def is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def pad_vocab_size(vocab_size: int, multiple: int = VOCAB_ALIGN) -> int:
    """Round vocab_size up to a multiple for Tensor Core / SIMD alignment."""
    if multiple <= 1:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None if missing, unreadable, or non-dict."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def read_json_dict(path: Path) -> dict[str, Any]:
    """Read a required JSON object, raising if missing, invalid, or non-dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


@contextmanager
def atomic_path(path: Path) -> Generator[Path]:
    """Yield a temporary sibling path and atomically replace ``path`` on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object to disk."""
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_safetensors_keys(weights_file: Path) -> set[str] | None:
    if not weights_file.is_file():
        return None
    try:
        with safe_open(str(weights_file), framework="pt", device="cpu") as f:
            return set(f.keys())
    except OSError, SafetensorError:
        return None


def infer_arch_from_weights(weights_file: Path) -> str | None:
    keys = read_safetensors_keys(weights_file)
    if keys is None:
        return None
    return (
        "gpt2"
        if ("transformer.wpe.weight" in keys or "wpe.weight" in keys)
        else "modern"
    )


def _infer_tokenizer_from_config(cfg: dict[str, Any]) -> str:
    tok = cfg.get("tokenizer")
    if isinstance(tok, str) and tok in VALID_TOKENIZERS:
        return tok
    init_from = cfg.get("init_from")
    if init_from in PRETRAINED_CONFIGS:
        return "gpt2"
    vocab_size = cfg.get("vocab_size")
    if (
        init_from == "scratch"
        and is_int(vocab_size)
        and vocab_size < BPE_MIN_VOCAB_SIZE
    ):
        return "char"
    return DEFAULT_TOKENIZER


def _infer_arch(init_from: str, ckpt_cfg: dict[str, Any] | None) -> str:
    if init_from in PRETRAINED_CONFIGS:
        return "gpt2"
    if init_from == "scratch":
        return DEFAULT_ARCH
    arch = infer_arch_from_weights(Path(init_from) / "model.safetensors")
    if arch is not None:
        return arch
    if ckpt_cfg is not None:
        cfg_arch = ckpt_cfg.get("arch")
        return (
            cfg_arch
            if isinstance(cfg_arch, str) and cfg_arch in VALID_ARCHS
            else "gpt2"
        )
    return DEFAULT_ARCH


def _base_block_size(init_from: str) -> int | None:
    if init_from in PRETRAINED_CONFIGS:
        return PRETRAINED_BLOCK_SIZE
    ckpt_cfg = (
        None if init_from == "scratch" else read_json(Path(init_from) / "config.json")
    )
    if ckpt_cfg is not None:
        block_size = ckpt_cfg.get("block_size")
        if is_int(block_size):
            return block_size
    return None


def _canonical_init_from(init_from: str) -> str:
    if init_from == "scratch" or init_from in PRETRAINED_CONFIGS:
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
    """Return baseline TrainConfig overrides for a validated target device."""
    if target == "cpu":
        return {"device": "cpu"}
    if target == "mps":
        mps_bf16 = hasattr(
            torch.backends.mps, "is_macos_or_newer"
        ) and torch.backends.mps.is_macos_or_newer(14, 0)
        return {
            **_GPU_PRESET,
            "device": "mps",
            "mixed_precision": "bf16" if mps_bf16 else "no",
            "batch_size": 32,
        }
    if target == "cuda":
        vram_gb = 0.0
        bf16 = False
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
            bf16 = torch.cuda.is_bf16_supported()
        if 0.0 < vram_gb < 12.0:
            batch_size, grad_accum, lr, beta2, max_iters = 32, 2, 1e-3, 0.99, 5000
        elif vram_gb >= 40.0:
            batch_size, grad_accum, lr, beta2, max_iters = 128, 1, 1.4e-3, 0.95, 2500
        else:
            batch_size, grad_accum, lr, beta2, max_iters = 64, 1, 1e-3, 0.99, 5000
        return {
            **_GPU_PRESET,
            "device": "cuda",
            "mixed_precision": "bf16" if bf16 else "fp16",
            "compile": True,
            "compile_mode": "reduce-overhead",
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
            "min_lr": lr * 0.1,
            "beta2": beta2,
            "max_iters": max_iters,
            "lr_decay_iters": max_iters,
        }
    raise ValueError(
        f"Unsupported device preset {target!r}; expected one of {sorted(VALID_DEVICES)}"
    )


@dataclass
class TrainConfig:
    """Hyperparameters, paths, and Accelerate runtime settings for GPT training."""

    data_path: Path = field(default_factory=lambda: _DEFAULT_ROOT / "data")
    batch_size: int = 16
    block_size: int = 128
    vocab_size: int = DEFAULT_BPE_VOCAB_SIZE
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    arch: str = DEFAULT_ARCH
    init_from: str = "scratch"
    tokenizer: str = DEFAULT_TOKENIZER
    bpe_vocab_size: int = DEFAULT_BPE_VOCAB_SIZE
    weight_decay: float = 0.1
    learning_rate: float = 1.5e-3
    min_lr: float = 1.5e-4
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    max_iters: int = 1000
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
        self.data_path = Path(self.data_path)
        for name, allowed in (
            ("device", VALID_DEVICES),
            ("mixed_precision", VALID_PRECISIONS),
            ("arch", VALID_ARCHS),
            ("compile_mode", VALID_COMPILE_MODES),
            ("tokenizer", VALID_TOKENIZERS),
        ):
            val = getattr(self, name)
            if val not in allowed:
                raise ValueError(
                    f"Unsupported {name} {val!r}; expected one of {sorted(allowed)}"
                )
        if self.device == "mps" and self.mixed_precision == "fp16":
            raise ValueError("MPS does not support fp16 autocast; use 'bf16' or 'no'")
        if self.init_from in PRETRAINED_CONFIGS:
            self.arch = "gpt2"
            self.tokenizer = "gpt2"

        int_bounds = (
            (
                1,
                (
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
                ),
            ),
            (0, ("warmup_iters", "lr_decay_iters", "early_stopping_patience")),
        )
        for min_val, names in int_bounds:
            for name in names:
                val = getattr(self, name)
                if not is_int(val) or val < min_val:
                    raise ValueError(
                        f"{name} must be an integer >= {min_val}, got {val!r}"
                    )
        if self.seed is not None and not is_int(self.seed):
            raise ValueError(f"seed must be an integer or None, got {self.seed!r}")

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
        if self.tokenizer == "bpe" and self.bpe_vocab_size < BPE_MIN_VOCAB_SIZE:
            raise ValueError(
                f"bpe_vocab_size must be >= {BPE_MIN_VOCAB_SIZE} "
                f"(256 byte tokens + <|endoftext|>), got {self.bpe_vocab_size}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip must be >= 0")
        if self.learning_rate <= 0 or not 0.0 <= self.min_lr <= self.learning_rate:
            raise ValueError(
                f"Expected 0 <= min_lr <= learning_rate and learning_rate > 0, "
                f"got min_lr={self.min_lr}, learning_rate={self.learning_rate}"
            )
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError(
                f"beta1 and beta2 must be in [0, 1), got {self.beta1}, {self.beta2}"
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
            "max_iters": int(self.max_iters),
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
                    max(int(self.vocab_size), GPT2_VOCAB_SIZE)
                    if int(self.vocab_size) >= GPT2_VOCAB_SIZE
                    else DEFAULT_GPT2_VOCAB_SIZE
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
        elif self.init_from in PRETRAINED_CONFIGS:
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
        out = {k: v for k, v in asdict(self).items() if k != "data_path"}
        if out["init_from"] != "scratch" and out["init_from"] not in PRETRAINED_CONFIGS:
            out["init_from"] = Path(str(out["init_from"])).name
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "arch" not in filtered:
            filtered["arch"] = "gpt2"
        if "tokenizer" not in filtered:
            filtered["tokenizer"] = _infer_tokenizer_from_config(filtered)
        if (
            filtered.get("tokenizer") == "bpe"
            and "bpe_vocab_size" not in filtered
            and is_int(filtered.get("vocab_size"))
            and filtered["vocab_size"] >= BPE_MIN_VOCAB_SIZE
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
        if (
            "learning_rate" in overrides
            and "min_lr" not in overrides
            and isinstance(overrides["learning_rate"], (int, float))
            and not isinstance(overrides["learning_rate"], bool)
        ):
            preset["min_lr"] = float(overrides["learning_rate"]) * 0.1

        init_from = str(overrides["init_from"]) if "init_from" in overrides else None
        ckpt_cfg = (
            read_json(Path(init_from) / "config.json")
            if init_from
            and init_from != "scratch"
            and init_from not in PRETRAINED_CONFIGS
            else None
        )
        if init_from is not None:
            if "arch" not in overrides:
                preset["arch"] = _infer_arch(init_from, ckpt_cfg)
            if "tokenizer" not in overrides:
                preset["tokenizer"] = (
                    "gpt2"
                    if init_from in PRETRAINED_CONFIGS
                    else (
                        _infer_tokenizer_from_config(ckpt_cfg)
                        if ckpt_cfg is not None
                        else DEFAULT_TOKENIZER
                    )
                )

        target_tokenizer = str(
            overrides.get("tokenizer", preset.get("tokenizer", DEFAULT_TOKENIZER))
        )
        if target_tokenizer == "bpe":
            if (
                ckpt_cfg is not None
                and ckpt_cfg.get("tokenizer") == "bpe"
                and "bpe_vocab_size" not in overrides
                and "vocab_size" not in overrides
            ):
                inferred = ckpt_cfg.get("bpe_vocab_size")
                if not is_int(inferred):
                    inferred = ckpt_cfg.get("vocab_size")
                if is_int(inferred) and inferred >= BPE_MIN_VOCAB_SIZE:
                    preset["bpe_vocab_size"] = inferred
                    preset["vocab_size"] = inferred
            elif "vocab_size" in overrides and "bpe_vocab_size" not in overrides:
                preset["bpe_vocab_size"] = overrides["vocab_size"]
            elif "bpe_vocab_size" in overrides and "vocab_size" not in overrides:
                preset["vocab_size"] = overrides["bpe_vocab_size"]
            elif "vocab_size" not in overrides:
                preset["vocab_size"] = preset.get(
                    "bpe_vocab_size", DEFAULT_BPE_VOCAB_SIZE
                )
        elif target_tokenizer == "gpt2" and "vocab_size" not in overrides:
            preset["vocab_size"] = DEFAULT_GPT2_VOCAB_SIZE

        preset.update(overrides)
        return cls(**preset)
