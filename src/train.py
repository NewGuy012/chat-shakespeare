from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DDPCommunicationHookType,
    DistributedDataParallelKwargs,
    DynamoBackend,
    TorchDynamoPlugin,
    set_seed,
)
from safetensors import safe_open
from safetensors.torch import load_model
from torch.utils.data import DataLoader, TensorDataset

from hyperparameters import (
    _GPT2_MIN_VOCAB_SIZE,
    TrainConfig,
    _read_safetensors_keys,
    pad_vocab_size,
)
from model import _PRETRAINED_CONFIGS, GPT, GPTConfig
from tokenizer import load_tokenizer, prepare_tokens

_MAX_DEVICE_TOKENS_BYTES = 256 * 1024 * 1024


def _configure_device_runtime(device: torch.device) -> None:
    """Enable device-appropriate fast math paths without affecting portability."""
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")


def _sync_device(device: torch.device) -> None:
    """Synchronize asynchronous device execution for accurate step timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()


def build_accelerator(config: TrainConfig) -> Accelerator:
    """Create Accelerator, honoring accelerate-launch environment overrides."""
    mixed_precision = (
        None if "ACCELERATE_MIXED_PRECISION" in os.environ else config.mixed_precision
    )
    effective_mp = os.environ.get("ACCELERATE_MIXED_PRECISION", mixed_precision)
    dynamo_plugin = None
    if "ACCELERATE_DYNAMO_BACKEND" not in os.environ and config.compile:
        effective_mode = config.compile_mode if config.device == "cuda" else "default"
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend.INDUCTOR,
            mode=effective_mode,
        )
    kwargs_handlers = None
    if config.device == "cuda":
        comm_wrapper = (
            DDPCommunicationHookType.BF16
            if effective_mp == "bf16"
            else DDPCommunicationHookType.NO
        )
        kwargs_handlers = [
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=True,
                comm_wrapper=comm_wrapper,
            )
        ]
    return Accelerator(
        cpu=(config.device == "cpu"),
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )


class TokenBatcher:
    """Vectorized on-device batch sampler avoiding per-step CPU-GPU transfers."""

    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        block_size: int,
        num_processes: int = 1,
        seed: int | None = None,
    ) -> None:
        if len(tokens) <= block_size:
            raise ValueError(
                f"Split length ({len(tokens)}) is too short for block_size={block_size}"
            )
        self.tokens = tokens
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_start = len(tokens) - block_size
        self.offsets = torch.arange(block_size, dtype=torch.long, device=tokens.device)
        n_blocks = (len(tokens) - 1) // block_size
        self.epoch_batches = max(1, n_blocks // max(1, batch_size * num_processes))
        self.seed = seed
        self._train_gen: torch.Generator | None = None
        if seed is not None:
            self._train_gen = torch.Generator(device=tokens.device).manual_seed(seed)

    def _sample_batch(
        self, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(
            0,
            self.max_start,
            (self.batch_size,),
            device=self.tokens.device,
            generator=generator,
        )
        idx = starts[:, None] + self.offsets[None, :]
        return self.tokens[idx], self.tokens[idx + 1]

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sample_batch(self._train_gen)

    def iter_eval(
        self, num_batches: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        eval_gen: torch.Generator | None = None
        if self.seed is not None:
            eval_gen = torch.Generator(device=self.tokens.device).manual_seed(self.seed)
        for _ in range(num_batches):
            yield self._sample_batch(eval_gen)


def _load_split_tensor(config: TrainConfig, split: str) -> torch.Tensor:
    file_path = config.train_path if split == "train" else config.val_path
    with safe_open(file_path, framework="pt", device="cpu") as f:
        return f.get_tensor(split)


def build_dataloader(
    config: TrainConfig,
    split: str,
    pin_memory: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Load split tokens on CPU and return a DataLoader of (x, y) context windows."""
    data = _load_split_tensor(config, split)
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
        drop_last=drop_last,
    )


def _build_batch_source(
    config: TrainConfig,
    accelerator: Accelerator,
    split: str,
    shuffle: bool,
) -> TokenBatcher | DataLoader:
    data = _load_split_tensor(config, split)
    if data.nbytes <= _MAX_DEVICE_TOKENS_BYTES:
        seed = (
            None
            if config.seed is None
            else config.seed
            + accelerator.process_index * 1000
            + (0 if split == "train" else 1)
        )
        tokens = data.to(device=accelerator.device, dtype=torch.long)
        return TokenBatcher(
            tokens,
            batch_size=config.batch_size,
            block_size=config.block_size,
            num_processes=accelerator.num_processes,
            seed=seed,
        )

    n_blocks = (len(data) - 1) // config.block_size
    drop_last = n_blocks >= config.batch_size * accelerator.num_processes
    loader = build_dataloader(
        config,
        split,
        pin_memory=(accelerator.device.type == "cuda"),
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return accelerator.prepare(loader)


def _cycle_batches(
    source: TokenBatcher | DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(source, TokenBatcher):
        while True:
            yield source.next_batch()
    epoch = 0
    while True:
        sampler = getattr(source, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        yield from source
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
    keys = _read_safetensors_keys(weights_file)
    if keys is not None:
        cfg.arch = (
            "gpt2"
            if ("transformer.wpe.weight" in keys or "wpe.weight" in keys)
            else "modern"
        )
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
    config.arch = model.config.arch


def initialize_model(config: TrainConfig) -> tuple[GPT, torch.optim.Optimizer]:
    """Initialize GPT (from scratch, HF GPT-2, or local checkpoint) and AdamW optimizer."""
    if config.init_from == "scratch":
        if config.tokenizer in {"bpe", "char"} and config.meta_path.exists():
            tok_vocab, _, _ = load_tokenizer(config.meta_path)
            config.vocab_size = pad_vocab_size(tok_vocab, 64)
        elif config.tokenizer == "gpt2":
            base_vocab = max(config.vocab_size, _GPT2_MIN_VOCAB_SIZE)
            config.vocab_size = pad_vocab_size(base_vocab, 64)
        else:
            config.vocab_size = pad_vocab_size(config.vocab_size, 64)
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
    loaders: dict[str, TokenBatcher | DataLoader],
) -> dict[str, float]:
    """Evaluate mean cross-entropy loss across train and val batch sources."""
    out: dict[str, float] = {}
    was_training = model.training
    model.eval()
    try:
        for split, loader in loaders.items():
            batch_losses: list[torch.Tensor] = []
            batch_iter: Iterable[tuple[torch.Tensor, torch.Tensor]] = (
                loader.iter_eval(config.eval_iters)
                if isinstance(loader, TokenBatcher)
                else loader
            )
            for step, (x, y) in enumerate(batch_iter):
                if step >= config.eval_iters:
                    break
                with accelerator.autocast():
                    _, loss = model(x, y, return_logits=False)
                batch_losses.append(loss.detach())
            if batch_losses:
                local_mean = torch.stack(batch_losses).mean()
                reduced = accelerator.reduce(local_mean, reduction="mean")
                out[split] = float(reduced.item())
            else:
                out[split] = float("nan")
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
    _configure_device_runtime(accelerator.device)
    if config.seed is not None:
        set_seed(config.seed)

    train_source = _build_batch_source(config, accelerator, "train", shuffle=True)
    val_source = _build_batch_source(config, accelerator, "val", shuffle=False)

    model, optimizer = initialize_model(config)
    accelerator.print(
        f"\nTraining on {accelerator.device} "
        f"(arch={model.config.arch}, mixed_precision={accelerator.mixed_precision}, "
        f"params={model.get_num_params() / 1e6:.2f}M)"
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    epoch_batches = (
        train_source.epoch_batches
        if isinstance(train_source, TokenBatcher)
        else len(train_source)
    )
    max_iters = (
        epoch_batches * config.epoch_iters
        if config.epoch_iters > 0
        else config.max_iters
    )
    accelerator.print(
        f"\t1 epoch = {epoch_batches} batches; running {max_iters} iterations."
    )

    losses: dict[str, float] = {}
    norm_tensor: torch.Tensor | None = None
    best_val = float("inf")
    no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_losses: dict[str, float] | None = None
    improved_last = False

    batch_iter = _cycle_batches(train_source)
    model.train()
    _sync_device(accelerator.device)
    t_log = time.perf_counter()
    last_log_iter = 0

    for iter_num in range(max_iters):
        x, y = next(batch_iter)
        with accelerator.accumulate(model):
            with accelerator.autocast():
                _, loss = model(x, y, return_logits=False)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if config.grad_clip > 0:
                    norm_tensor = accelerator.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                lr = get_lr(config, iter_num)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if iter_num % config.log_iters == 0 or iter_num == max_iters - 1:
            _sync_device(accelerator.device)
            dt = max(time.perf_counter() - t_log, 1e-9)
            norm = float(norm_tensor.item()) if norm_tensor is not None else 0.0
            losses = estimate_loss(
                config,
                accelerator,
                model,
                {"train": train_source, "val": val_source},
            )
            timing = ""
            if iter_num > 0:
                steps = iter_num - last_log_iter
                ms_per_iter = dt / steps * 1000.0
                tok_per_s = (
                    steps
                    * config.batch_size
                    * config.block_size
                    * accelerator.num_processes
                    / dt
                )
                timing = f", {ms_per_iter:.1f} ms/iter, {tok_per_s:,.0f} tok/s"
            accelerator.print(
                f"\tStep {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, norm {norm:.4f}{timing}"
            )

            if config.early_stopping_patience > 0:
                val_loss = losses["val"]
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    improved_last = True
                    best_losses = dict(losses)
                    unwrapped = accelerator.unwrap_model(
                        model, keep_torch_compile=False
                    )
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in unwrapped.state_dict().items()
                    }
                else:
                    improved_last = False
                    no_improve += 1
                    if no_improve >= config.early_stopping_patience:
                        accelerator.print(
                            f"\tEarly stopping at step {iter_num} "
                            f"(best val loss {best_val:.4f})."
                        )
                        break

            _sync_device(accelerator.device)
            t_log = time.perf_counter()
            last_log_iter = iter_num

    if best_state is not None and best_losses is not None and not improved_last:
        accelerator.unwrap_model(model, keep_torch_compile=False).load_state_dict(
            best_state, strict=True
        )
        losses = best_losses

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
    parser.add_argument("--arch", choices=("modern", "gpt2"), default=None)
    parser.add_argument("--init-from", default="scratch")
    parser.add_argument("--tokenizer", choices=("bpe", "char", "gpt2"), default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default=None,
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, object] = {"init_from": args.init_from}
    if args.arch is not None:
        overrides["arch"] = args.arch
    if args.tokenizer is not None:
        overrides["tokenizer"] = args.tokenizer
    if args.vocab_size is not None:
        overrides["vocab_size"] = args.vocab_size
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["gradient_accumulation_steps"] = args.grad_accum
    if args.block_size is not None:
        overrides["block_size"] = args.block_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
        overrides["min_lr"] = args.lr * 0.1
    if args.max_iters is not None:
        overrides["max_iters"] = args.max_iters
    if args.compile_mode is not None:
        overrides["compile_mode"] = args.compile_mode
    if args.no_compile:
        overrides["compile"] = False
    if args.early_stopping is not None:
        overrides["early_stopping_patience"] = args.early_stopping
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
