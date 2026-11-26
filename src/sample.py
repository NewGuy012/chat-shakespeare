from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from hyperparameters import GPT2_VOCAB_SIZE, TrainConfig, is_int
from model import GPT
from tokenizer import load_tokenizer
from train import load_checkpoint

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class SampleConfig:
    start: str = "\n"
    num_samples: int = 1
    max_new_tokens: int = 500
    temperature: float = 0.95
    top_k: int | None = 200

    def __post_init__(self) -> None:
        if not is_int(self.num_samples) or self.num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {self.num_samples!r}")
        if not is_int(self.max_new_tokens) or self.max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be >= 0, got {self.max_new_tokens!r}"
            )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k is not None and (not is_int(self.top_k) or self.top_k < 1):
            raise ValueError(f"top_k must be None or >= 1, got {self.top_k!r}")


def sample(
    config: TrainConfig,
    sample_config: SampleConfig,
    model: GPT | None = None,
) -> list[str]:
    """Generate text samples from a saved checkpoint or pre-loaded GPT model."""
    if sample_config.num_samples == 0:
        return []

    device = torch.device(config.device)
    model_passed = model is not None
    if model is None:
        ckpt_cfg, model = load_checkpoint(config.checkpoint_dir, device=device)
    else:
        ckpt_cfg = config
        model.to(device)

    if ckpt_cfg.tokenizer == "gpt2" or (
        model_passed and model.config.vocab_size >= GPT2_VOCAB_SIZE
    ):
        meta_path = None
    else:
        meta_path = (
            ckpt_cfg.meta_path if ckpt_cfg.meta_path.is_file() else config.meta_path
        )
    tok_vocab_size, encode, decode = load_tokenizer(meta_path)

    start_ids = encode(sample_config.start)
    if not start_ids:
        raise ValueError("Prompt encoded to zero tokens")
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    if sample_config.num_samples > 1:
        x = x.expand(sample_config.num_samples, -1)

    amp_dtype = _AMP_DTYPES.get(config.mixed_precision)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type in {"cuda", "mps"}
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
