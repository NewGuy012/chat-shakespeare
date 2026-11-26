from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Iterator
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

from hyperparameters import (
    GPT2_VOCAB_SIZE,
    PRETRAINED_CONFIGS,
    VALID_ARCHS,
    VALID_COMPILE_MODES,
    VALID_DEVICES,
    VALID_TOKENIZERS,
    VOCAB_ALIGN,
    TrainConfig,
    infer_arch_from_weights,
    is_int,
    pad_vocab_size,
    read_json,
    read_json_dict,
    read_safetensors_keys,
    write_json_atomic,
)
from model import GPT, GPTConfig
from tokenizer import prepare_tokens

_MAX_DEVICE_TOKENS_BYTES = 256 * 1024 * 1024


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_accelerator(config: TrainConfig) -> Accelerator:
    """Create Accelerator, honoring accelerate-launch environment overrides."""
    mixed_precision = (
        None if "ACCELERATE_MIXED_PRECISION" in os.environ else config.mixed_precision
    )
    effective_mp = os.environ.get("ACCELERATE_MIXED_PRECISION", mixed_precision)
    dynamo_plugin = None
    if "ACCELERATE_DYNAMO_BACKEND" not in os.environ and config.compile:
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend.INDUCTOR,
            mode=config.compile_mode if config.device == "cuda" else "default",
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
    """Vectorized batch sampler keeping small splits on-device and large splits on CPU."""

    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        block_size: int,
        target_device: torch.device,
        num_processes: int = 1,
        seed: int | None = None,
    ) -> None:
        if len(tokens) <= block_size:
            raise ValueError(
                f"Split length ({len(tokens)}) is too short for block_size={block_size}"
            )
        self.on_device = tokens.nbytes <= _MAX_DEVICE_TOKENS_BYTES
        self.tokens = (
            tokens.to(device=target_device, dtype=torch.long)
            if self.on_device
            else tokens
        )
        self.target_device = target_device
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_start = len(tokens) - block_size
        self.offsets = torch.arange(
            block_size, dtype=torch.long, device=self.tokens.device
        )
        n_blocks = (len(tokens) - 1) // block_size
        self.epoch_batches = max(1, n_blocks // max(1, batch_size * num_processes))
        self.seed = seed
        self._train_gen = (
            torch.Generator(device=self.tokens.device).manual_seed(seed)
            if seed is not None
            else None
        )

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
        x = self.tokens[idx]
        y = self.tokens[idx + 1]
        if not self.on_device:
            non_blocking = self.target_device.type == "cuda"
            x = x.to(
                device=self.target_device, dtype=torch.long, non_blocking=non_blocking
            )
            y = y.to(
                device=self.target_device, dtype=torch.long, non_blocking=non_blocking
            )
        return x, y

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sample_batch(self._train_gen)

    def iter_eval(
        self, num_batches: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        eval_gen = (
            torch.Generator(device=self.tokens.device).manual_seed(self.seed)
            if self.seed is not None
            else None
        )
        for _ in range(num_batches):
            yield self._sample_batch(eval_gen)


def _build_batcher(
    config: TrainConfig, accelerator: Accelerator, split: str
) -> TokenBatcher:
    file_path = config.train_path if split == "train" else config.val_path
    with safe_open(file_path, framework="pt", device="cpu") as f:
        data = f.get_tensor(split)
    seed = (
        None
        if config.seed is None
        else config.seed
        + accelerator.process_index * 1000
        + (0 if split == "train" else 1)
    )
    return TokenBatcher(
        data,
        batch_size=config.batch_size,
        block_size=config.block_size,
        target_device=accelerator.device,
        num_processes=accelerator.num_processes,
        seed=seed,
    )


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
    if not config_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint in {checkpoint_dir}")

    saved = read_json_dict(config_file)
    arch = infer_arch_from_weights(weights_file)
    if arch is None:
        raise ValueError(f"Invalid safetensors weights in {checkpoint_dir}")
    saved["arch"] = arch
    if "data_path" not in saved and checkpoint_dir.parent.name == "checkpoints":
        saved["data_path"] = checkpoint_dir.parent.parent

    cfg = TrainConfig.from_dict(saved)
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
    for field_name in (
        "block_size",
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
        "bias",
        "arch",
    ):
        setattr(config, field_name, getattr(model.config, field_name))


def initialize_model(config: TrainConfig) -> tuple[GPT, torch.optim.Optimizer]:
    """Initialize GPT (from scratch, HF GPT-2, or local checkpoint) and AdamW optimizer."""
    if config.init_from == "scratch":
        if config.tokenizer in {"bpe", "char"} and config.meta_path.exists():
            meta = read_json_dict(config.meta_path)
            tok_vocab = meta.get("vocab_size")
            if not is_int(tok_vocab) or tok_vocab < 1:
                raise ValueError(
                    f"Invalid vocab_size in {config.meta_path}: {tok_vocab!r}"
                )
            config.vocab_size = pad_vocab_size(tok_vocab, VOCAB_ALIGN)
        elif config.tokenizer == "gpt2":
            config.vocab_size = pad_vocab_size(
                max(config.vocab_size, GPT2_VOCAB_SIZE), VOCAB_ALIGN
            )
        else:
            config.vocab_size = pad_vocab_size(config.vocab_size, VOCAB_ALIGN)
        model = GPT(GPTConfig(**config.model_kwargs()))
    else:
        if config.init_from in PRETRAINED_CONFIGS:
            model = GPT.from_pretrained(config.init_from, dropout=config.dropout)
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
    loaders: dict[str, TokenBatcher],
) -> dict[str, float]:
    """Evaluate mean cross-entropy loss across train and val batchers."""
    out: dict[str, float] = {}
    was_training = model.training
    model.eval()
    try:
        for split, loader in loaders.items():
            batch_losses: list[torch.Tensor] = []
            for x, y in loader.iter_eval(config.eval_iters):
                with accelerator.autocast():
                    _, loss = model(x, y, return_logits=False)
                batch_losses.append(loss.detach())
            local_mean = torch.stack(batch_losses).mean()
            out[split] = float(accelerator.reduce(local_mean, reduction="mean").item())
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
    if accelerator.device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if config.seed is not None:
        set_seed(config.seed)

    train_source = _build_batcher(config, accelerator, "train")
    val_source = _build_batcher(config, accelerator, "val")

    model, optimizer = initialize_model(config)
    accelerator.print(
        f"\nTraining on {accelerator.device} "
        f"(arch={model.config.arch}, mixed_precision={accelerator.mixed_precision}, "
        f"params={model.get_num_params() / 1e6:.2f}M)"
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.print(
        f"\t1 epoch = {train_source.epoch_batches} batches; running {config.max_iters} iterations."
    )

    losses: dict[str, float] = {}
    norm_tensor: torch.Tensor | None = None
    best_val = float("inf")
    no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_losses: dict[str, float] | None = None
    improved_last = False

    model.train()
    _sync_device(accelerator.device)
    t_log = time.perf_counter()
    last_log_iter = 0

    for iter_num in range(config.max_iters):
        x, y = train_source.next_batch()
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

        if iter_num % config.log_iters == 0 or iter_num == config.max_iters - 1:
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
    """Save unwrapped safetensors weights and atomic JSON config via Accelerate."""
    target_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else config.checkpoint_dir
    )
    unwrapped = accelerator.unwrap_model(model, keep_torch_compile=False)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        target_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_model(unwrapped, target_dir, safe_serialization=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        meta = {
            **config.to_dict(),
            "fingerprint": config.fingerprint(),
            "losses": losses or {},
        }
        write_json_atomic(target_dir / "config.json", meta)
    accelerator.wait_for_everyone()
    return target_dir


def checkpoint_ready(
    config: TrainConfig,
    checkpoint_dir: Path | None = None,
) -> bool:
    """Return True when valid weights and matching config.json exist."""
    target_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else config.checkpoint_dir
    )
    keys = read_safetensors_keys(target_dir / "model.safetensors")
    saved = read_json(target_dir / "config.json")
    if not keys or saved is None:
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
    parser.add_argument("--device", choices=VALID_DEVICES, default=None)
    parser.add_argument("--arch", choices=VALID_ARCHS, default=None)
    parser.add_argument("--init-from", default="scratch")
    parser.add_argument("--tokenizer", choices=VALID_TOKENIZERS, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--compile-mode", choices=VALID_COMPILE_MODES, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        args.init_from != "scratch"
        and args.init_from not in PRETRAINED_CONFIGS
        and not Path(args.init_from).is_dir()
    ):
        parser.error(f"Unknown --init-from checkpoint path: {args.init_from!r}")

    arg_map = {
        "arch": "arch",
        "tokenizer": "tokenizer",
        "vocab_size": "vocab_size",
        "batch_size": "batch_size",
        "grad_accum": "gradient_accumulation_steps",
        "block_size": "block_size",
        "lr": "learning_rate",
        "max_iters": "max_iters",
        "compile_mode": "compile_mode",
        "early_stopping": "early_stopping_patience",
        "data_path": "data_path",
    }
    overrides: dict[str, object] = {"init_from": args.init_from}
    for arg_name, field_name in arg_map.items():
        val = getattr(args, arg_name)
        if val is not None:
            overrides[field_name] = val
    if args.no_compile:
        overrides["compile"] = False

    config = TrainConfig.auto(device=args.device, **overrides)
    accelerator = build_accelerator(config)

    if accelerator.is_main_process:
        prepare_tokens(config, overwrite=args.overwrite)
    accelerator.wait_for_everyone()

    ensure_checkpoint(config, accelerator=accelerator, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
