from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from accelerate import PartialState

from hyperparameters import TrainConfig
from model import GPT
from tokenizer import load_tokenizer
from train import load_checkpoint


@dataclass
class SampleConfig:
    start: str = "\n"
    num_samples: int = 1
    max_new_tokens: int = 500
    temperature: float = 0.95
    top_k: int | None = 200


def sample(
    config: TrainConfig,
    sample_config: SampleConfig,
    model: GPT | None = None,
) -> list[str]:
    """Generate text samples from a saved checkpoint or pre-loaded GPT model."""
    if sample_config.num_samples <= 0:
        return []

    state = PartialState(cpu=(config.device == "cpu"))
    device = state.device if state.device is not None else state.default_device

    if model is None:
        ckpt_cfg, model = load_checkpoint(config.checkpoint_dir, device=device)
        meta_path = (
            ckpt_cfg.meta_path if ckpt_cfg.meta_path.exists() else config.meta_path
        )
    else:
        model.to(device)
        meta_path = config.meta_path if config.meta_path.exists() else None

    tok_vocab_size, encode, decode = load_tokenizer(meta_path)
    if tok_vocab_size < 50257 and model.config.vocab_size >= 50257:
        # Pretrained GPT-2 models expect GPT-2 BPE even if local meta.json is bpe or char.
        tok_vocab_size, encode, decode = load_tokenizer(None)

    start_ids = encode(sample_config.start)
    if not start_ids:
        raise ValueError("Prompt encoded to zero tokens")
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    if sample_config.num_samples > 1:
        x = x.expand(sample_config.num_samples, -1)

    amp_dtype = (
        torch.bfloat16
        if config.mixed_precision == "bf16"
        else (torch.float16 if config.mixed_precision == "fp16" else None)
    )
    use_amp = (
        amp_dtype is not None
        and device.type in {"cuda", "mps"}
        and not (device.type == "mps" and amp_dtype == torch.float16)
    )
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if use_amp and amp_dtype is not None
        else nullcontext()
    )

    outputs: list[str] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), amp_ctx:
            y = model.generate(
                x,
                sample_config.max_new_tokens,
                temperature=sample_config.temperature,
                top_k=sample_config.top_k,
                valid_vocab_size=tok_vocab_size,
            )
            for row in y.tolist():
                text = decode(row)
                outputs.append(text)
                print(text)
                print("---------------")
    finally:
        if was_training:
            model.train()
    return outputs
