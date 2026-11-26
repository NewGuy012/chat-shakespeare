from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from safetensors.torch import load_model
from torch.utils.data import DataLoader, TensorDataset

from hyperparameters import _DEFAULT_GPT2_VOCAB_SIZE, _GPT2_MIN_VOCAB_SIZE, TrainConfig
from model import _PRETRAINED_CONFIGS, GPT, GPTConfig
from tokenizer import load_tokenizer, prepare_tokens


def build_accelerator(config: TrainConfig) -> Accelerator:
    """Create Accelerator, honoring accelerate-launch environment overrides."""
    mixed_precision = (
        None if "ACCELERATE_MIXED_PRECISION" in os.environ else config.mixed_precision
    )
    dynamo_backend = (
        None
        if "ACCELERATE_DYNAMO_BACKEND" in os.environ
        else ("inductor" if config.compile else None)
    )
    return Accelerator(
        cpu=(config.device == "cpu"),
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dynamo_backend=dynamo_backend,
    )


def build_dataloader(
    config: TrainConfig,
    split: str,
    pin_memory: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Load split tokens on CPU and return a DataLoader of (x, y) context windows."""
    file_path = config.train_path if split == "train" else config.val_path
    with safe_open(file_path, framework="pt", device="cpu") as f:
        data = f.get_tensor(split)

    n_blocks = (len(data) - 1) // config.block_size
    if n_blocks < 1:
        raise ValueError(
            f"Split {split!r} length ({len(data)}) is too short for block_size={config.block_size}"
        )

    n_tokens = n_blocks * config.block_size
    x = data[:n_tokens].view(n_blocks, config.block_size)
    y = data[1 : n_tokens + 1].view(n_blocks, config.block_size)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def _cycle_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        yield from loader
        epoch += 1


def get_lr(config: TrainConfig, step: int) -> float:
    """Cosine learning rate decay with linear warmup."""
    if config.warmup_iters > 0 and step < config.warmup_iters:
        return config.learning_rate * (step + 1) / (config.warmup_iters + 1)
    if config.lr_decay_iters == 0:
        return config.learning_rate
    if step >= config.lr_decay_iters or config.lr_decay_iters <= config.warmup_iters:
        return config.min_lr

    decay_ratio = (step - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def load_checkpoint(
    checkpoint_dir: Path,
    device: str | torch.device = "cpu",
    dropout: float | None = None,
) -> tuple[TrainConfig, GPT]:
    """Load TrainConfig and GPT weights from a safetensors checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    config_file = checkpoint_dir / "config.json"
    weights_file = checkpoint_dir / "model.safetensors"
    if not config_file.exists() or not weights_file.exists():
        raise FileNotFoundError(f"Incomplete checkpoint in {checkpoint_dir}")

    cfg = TrainConfig.from_dict(json.loads(config_file.read_text(encoding="utf-8")))
    if dropout is not None:
        cfg.dropout = dropout
    model = GPT(GPTConfig(**cfg.model_kwargs()))
    missing, unexpected = load_model(model, str(weights_file), device="cpu")
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint mismatch in {checkpoint_dir}: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    return cfg, model


def _sync_config_from_model(config: TrainConfig, model: GPT) -> None:
    config.block_size = model.config.block_size
    config.vocab_size = model.config.vocab_size
    config.n_layer = model.config.n_layer
    config.n_head = model.config.n_head
    config.n_embd = model.config.n_embd
    config.bias = model.config.bias


def initialize_model(config: TrainConfig) -> tuple[GPT, torch.optim.Optimizer]:
    """Initialize GPT (from scratch, HF GPT-2, or local checkpoint) and AdamW optimizer."""
    if config.init_from == "scratch":
        if config.tokenizer in {"bpe", "char"} and config.meta_path.exists():
            config.vocab_size, _, _ = load_tokenizer(config.meta_path)
        elif config.tokenizer == "gpt2" and config.vocab_size < _GPT2_MIN_VOCAB_SIZE:
            config.vocab_size = _DEFAULT_GPT2_VOCAB_SIZE
        model = GPT(GPTConfig(**config.model_kwargs()))
    elif config.init_from in _PRETRAINED_CONFIGS:
        model = GPT.from_pretrained(config.init_from, dropout=config.dropout)
        if config.block_size < model.config.block_size:
            model.crop_block_size(config.block_size)
        _sync_config_from_model(config, model)
    else:
        ckpt_dir = Path(config.init_from)
        if not ckpt_dir.is_dir():
            raise ValueError(f"Unknown init_from value {config.init_from!r}")
        _, model = load_checkpoint(ckpt_dir, dropout=config.dropout)
        if config.block_size < model.config.block_size:
            model.crop_block_size(config.block_size)
        _sync_config_from_model(config, model)

    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, (config.beta1, config.beta2)
    )
    return model, optimizer


@torch.no_grad()
def estimate_loss(
    config: TrainConfig,
    accelerator: Accelerator,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
) -> dict[str, float]:
    """Evaluate mean cross-entropy loss on prepared train and val DataLoaders."""
    out: dict[str, float] = {}
    was_training = model.training
    model.eval()
    try:
        for split, loader in loaders.items():
            batch_losses: list[torch.Tensor] = []
            for step, (x, y) in enumerate(loader):
                if step >= config.eval_iters:
                    break
                with accelerator.autocast():
                    _, loss = model(x, y)
                batch_losses.append(accelerator.reduce(loss.detach(), reduction="mean"))
            out[split] = (
                float(torch.stack(batch_losses).mean().item())
                if batch_losses
                else float("nan")
            )
    finally:
        if was_training:
            model.train()
    return out


def train(
    config: TrainConfig,
    accelerator: Accelerator | None = None,
) -> tuple[Accelerator, torch.nn.Module, dict[str, float]]:
    """Train GPT with Hugging Face Accelerate across CPU, MPS, or single/multi-GPU."""
    accelerator = accelerator or build_accelerator(config)
    if config.seed is not None:
        set_seed(config.seed)

    pin_memory = accelerator.device.type == "cuda"
    train_loader = build_dataloader(
        config, "train", pin_memory=pin_memory, shuffle=True
    )
    val_loader = build_dataloader(config, "val", pin_memory=pin_memory, shuffle=False)

    model, optimizer = initialize_model(config)
    accelerator.print(
        f"\nTraining on {accelerator.device} "
        f"(mixed_precision={accelerator.mixed_precision}, params={model.get_num_params() / 1e6:.2f}M)"
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    epoch_batches = len(train_loader)
    max_iters = (
        epoch_batches * config.epoch_iters
        if config.epoch_iters > 0
        else config.max_iters
    )
    accelerator.print(
        f"\t1 epoch = {epoch_batches} batches; running {max_iters} iterations."
    )

    losses: dict[str, float] = {}
    norm = 0.0
    batch_iter = _cycle_batches(train_loader)
    model.train()

    for iter_num in range(max_iters):
        x, y = next(batch_iter)
        with accelerator.accumulate(model):
            with accelerator.autocast():
                _, loss = model(x, y)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if config.grad_clip > 0:
                    norm = float(
                        accelerator.clip_grad_norm_(
                            model.parameters(), config.grad_clip
                        )
                    )
                lr = get_lr(config, iter_num)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if iter_num % config.log_iters == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(
                config, accelerator, model, {"train": train_loader, "val": val_loader}
            )
            accelerator.print(
                f"\tStep {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, norm {norm:.4f}"
            )

    return accelerator, model, losses


def save_checkpoint(
    config: TrainConfig,
    accelerator: Accelerator,
    model: torch.nn.Module,
    losses: dict[str, float] | None = None,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Save unwrapped safetensors weights and JSON config via Accelerate."""
    target_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else config.checkpoint_dir
    )
    unwrapped = accelerator.unwrap_model(model, keep_torch_compile=False)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        target_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_model(unwrapped, target_dir, safe_serialization=True)
    if accelerator.is_main_process:
        meta = {
            **config.to_dict(),
            "fingerprint": config.fingerprint(),
            "losses": losses or {},
        }
        (target_dir / "config.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    return target_dir


def checkpoint_ready(
    config: TrainConfig,
    checkpoint_dir: Path | None = None,
) -> bool:
    """Return True when a complete checkpoint matching config's fingerprint exists."""
    target_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else config.checkpoint_dir
    )
    weights_file = target_dir / "model.safetensors"
    config_file = target_dir / "config.json"
    if not weights_file.is_file() or not config_file.is_file():
        return False
    try:
        saved = json.loads(config_file.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return False
    if not isinstance(saved, dict):
        return False
    saved_fp = saved.get("fingerprint")
    if isinstance(saved_fp, str):
        return saved_fp == config.fingerprint()
    try:
        return TrainConfig.from_dict(saved).fingerprint() == config.fingerprint()
    except TypeError, ValueError:
        return False


def ensure_checkpoint(
    config: TrainConfig,
    accelerator: Accelerator | None = None,
    overwrite: bool = False,
) -> Path:
    """Train and save a checkpoint unless a matching cached checkpoint already exists."""
    if not overwrite and checkpoint_ready(config):
        msg = f"Using cached checkpoint: {config.checkpoint_dir}"
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)
        return config.checkpoint_dir

    acc, model, losses = train(config, accelerator=accelerator)
    return save_checkpoint(config, acc, model, losses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Chat Shakespeare with Accelerate"
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--init-from", default="scratch")
    parser.add_argument("--tokenizer", choices=("bpe", "char", "gpt2"), default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, object] = {"init_from": args.init_from}
    if args.tokenizer is not None:
        overrides["tokenizer"] = args.tokenizer
    if args.vocab_size is not None:
        overrides["vocab_size"] = args.vocab_size
    if args.max_iters is not None:
        overrides["max_iters"] = args.max_iters
    if args.data_path is not None:
        overrides["data_path"] = args.data_path
    config = TrainConfig.auto(device=args.device, **overrides)
    accelerator = build_accelerator(config)

    if accelerator.is_main_process:
        prepare_tokens(config, overwrite=args.overwrite)
    accelerator.wait_for_everyone()

    ensure_checkpoint(config, accelerator=accelerator, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
