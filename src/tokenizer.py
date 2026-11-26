from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from safetensors.torch import save_file

from hyperparameters import TrainConfig


@lru_cache(maxsize=1)
def _gpt2_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def _split_texts(ds: DatasetDict | dict[str, Any]) -> tuple[list[str], list[str]]:
    val_split = "validation" if "validation" in ds else "test"
    return list(ds["train"]["text"]), list(ds[val_split]["text"])


def download(config: TrainConfig) -> DatasetDict:
    """Download or load cached tiny-shakespeare dataset."""
    if config.download_path.exists():
        return load_from_disk(config.download_path)

    dataset = load_dataset("Trelis/tiny-shakespeare")
    if "Text" in dataset["train"].column_names:
        dataset = dataset.rename_column("Text", "text")
    config.download_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(config.download_path)
    return dataset


def save_meta(config: TrainConfig, meta: dict[str, Any]) -> None:
    """Write tokenizer metadata as JSON (pickle-free)."""
    config.meta_path.parent.mkdir(parents=True, exist_ok=True)
    config.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def tokenize_char(config: TrainConfig, ds: DatasetDict | dict[str, Any]) -> dict[str, list[int]]:
    """Character-level tokenize train/val splits and persist metadata."""
    train_items, val_items = _split_texts(ds)
    train_text = "".join(train_items)
    val_text = "".join(val_items)
    chars = sorted(set(train_text + val_text))
    if not chars:
        raise ValueError("Cannot build character vocabulary from empty dataset")

    stoi = {c: i for i, c in enumerate(chars)}
    config.vocab_size = len(chars)
    print(f"\nTokenize text (char): vocab_size={config.vocab_size:,}")

    save_meta(
        config,
        {
            "tokenizer": "char",
            "vocab_size": config.vocab_size,
            "chars": "".join(chars),
        },
    )
    return {
        "train": [stoi[c] for c in train_text],
        "val": [stoi[c] for c in val_text],
    }


def tokenize_gpt2(config: TrainConfig, ds: DatasetDict | dict[str, Any]) -> dict[str, list[int]]:
    """Tokenize train/val splits with GPT-2 BPE (tiktoken) and persist metadata."""
    enc = _gpt2_encoding()
    train_items, val_items = _split_texts(ds)

    def encode_split(items: list[str]) -> list[int]:
        ids: list[int] = []
        for text in items:
            ids.extend(enc.encode_ordinary(text))
            ids.append(enc.eot_token)
        return ids

    vocab_size = max(config.vocab_size, enc.n_vocab)
    config.vocab_size = vocab_size
    print(f"\nTokenize text (gpt2): vocab_size={vocab_size:,}")

    save_meta(
        config,
        {
            "tokenizer": "gpt2",
            "vocab_size": vocab_size,
        },
    )
    return {
        "train": encode_split(train_items),
        "val": encode_split(val_items),
    }


def save_tensors(config: TrainConfig, tokenized: dict[str, list[int]]) -> None:
    """Persist tokenized train and val splits as safetensors."""
    config.data_path.mkdir(parents=True, exist_ok=True)
    save_file({"train": torch.tensor(tokenized["train"], dtype=torch.long)}, config.train_path)
    save_file({"val": torch.tensor(tokenized["val"], dtype=torch.long)}, config.val_path)


def load_tokenizer(
    meta_path: Path | None = None,
) -> tuple[int, Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Return (vocab_size, encode, decode) from meta.json or default GPT-2 BPE."""
    if meta_path is not None and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("tokenizer") == "char":
            chars = meta["chars"]
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}

            def encode_char(text: str) -> list[int]:
                unknown = sorted({c for c in text if c not in stoi})
                if unknown:
                    raise ValueError(f"Unsupported character(s) for char tokenizer: {unknown}")
                return [stoi[c] for c in text]

            def decode_char(ids: list[int]) -> str:
                return "".join(itos[i] for i in ids)

            return len(chars), encode_char, decode_char

        vocab_size = int(meta.get("vocab_size", 50304))
    else:
        vocab_size = 50304

    enc = _gpt2_encoding()

    def encode_gpt2(text: str) -> list[int]:
        return enc.encode(text, allowed_special={"<|endoftext|>"})

    return vocab_size, encode_gpt2, enc.decode
