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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from hyperparameters import _DEFAULT_GPT2_VOCAB_SIZE, TrainConfig

_EOT_TOKEN = "<|endoftext|>"
_BPE_TOKENIZER_FILE = "tokenizer.json"


@lru_cache(maxsize=1)
def _gpt2_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def _split_texts(ds: DatasetDict | dict[str, Any]) -> tuple[list[str], list[str]]:
    val_split = "validation" if "validation" in ds else "test"
    return list(ds["train"]["text"]), list(ds[val_split]["text"])


def download(config: TrainConfig) -> DatasetDict:
    """Download or load cached tiny-shakespeare dataset."""
    if config.download_path.exists():
        loaded = load_from_disk(config.download_path)
        if not isinstance(loaded, DatasetDict):
            raise TypeError(
                f"Expected DatasetDict at {config.download_path}, got {type(loaded).__name__}"
            )
        return loaded

    dataset = load_dataset("Trelis/tiny-shakespeare")
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict from load_dataset, got {type(dataset).__name__}"
        )
    if "Text" in dataset["train"].column_names:
        dataset = dataset.rename_column("Text", "text")
    config.download_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(config.download_path)
    return dataset


def save_meta(config: TrainConfig, meta: dict[str, Any]) -> None:
    """Write tokenizer metadata as JSON (pickle-free)."""
    config.meta_path.parent.mkdir(parents=True, exist_ok=True)
    config.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def tokenize_char(
    config: TrainConfig, ds: DatasetDict | dict[str, Any]
) -> dict[str, list[int]]:
    """Character-level tokenize train/val splits and persist metadata."""
    train_items, val_items = _split_texts(ds)
    train_text = "".join(train_items)
    val_text = "".join(val_items)
    chars = sorted(set(train_text + val_text))
    if not chars:
        raise ValueError("Cannot build character vocabulary from empty dataset")

    stoi = {c: i for i, c in enumerate(chars)}
    config.tokenizer = "char"
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


def tokenize_bpe(
    config: TrainConfig, ds: DatasetDict | dict[str, Any]
) -> dict[str, list[int]]:
    """Train a byte-level BPE tokenizer and tokenize train/val splits."""
    train_items, val_items = _split_texts(ds)
    if not any(train_items):
        raise ValueError("Cannot train BPE tokenizer from empty training dataset")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.bpe_vocab_size,
        special_tokens=[_EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(train_items, trainer=trainer)

    eot_id = tok.token_to_id(_EOT_TOKEN)
    if eot_id is None:
        raise RuntimeError(f"Missing special token {_EOT_TOKEN!r} after BPE training")

    def encode_split(items: list[str]) -> list[int]:
        ids: list[int] = []
        for encoding in tok.encode_batch(items):
            ids.extend(encoding.ids)
            ids.append(eot_id)
        return ids

    actual_vocab_size = tok.get_vocab_size()
    config.tokenizer = "bpe"
    config.vocab_size = actual_vocab_size
    print(
        f"\nTokenize text (bpe): vocab_size={actual_vocab_size:,} "
        f"(target={config.bpe_vocab_size:,})"
    )

    config.token_dir.mkdir(parents=True, exist_ok=True)
    tok_path = config.token_dir / _BPE_TOKENIZER_FILE
    tok.save(str(tok_path))
    save_meta(
        config,
        {
            "tokenizer": "bpe",
            "vocab_size": actual_vocab_size,
            "bpe_vocab_size": config.bpe_vocab_size,
            "tokenizer_file": _BPE_TOKENIZER_FILE,
        },
    )
    return {
        "train": encode_split(train_items),
        "val": encode_split(val_items),
    }


def tokenize_gpt2(
    config: TrainConfig, ds: DatasetDict | dict[str, Any]
) -> dict[str, list[int]]:
    """Tokenize train/val splits with GPT-2 BPE (tiktoken) and persist metadata."""
    enc = _gpt2_encoding()
    train_items, val_items = _split_texts(ds)

    def encode_split(items: list[str]) -> list[int]:
        ids: list[int] = []
        for text in items:
            ids.extend(enc.encode_ordinary(text))
            ids.append(enc.eot_token)
        return ids

    vocab_size = (
        config.vocab_size
        if config.vocab_size >= enc.n_vocab
        else _DEFAULT_GPT2_VOCAB_SIZE
    )
    config.tokenizer = "gpt2"
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
    config.token_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"train": torch.tensor(tokenized["train"], dtype=torch.long)}, config.train_path
    )
    save_file(
        {"val": torch.tensor(tokenized["val"], dtype=torch.long)}, config.val_path
    )


def tokens_ready(config: TrainConfig) -> bool:
    """Return True when tokenized splits and matching meta.json exist on disk."""
    if not (
        config.train_path.is_file()
        and config.val_path.is_file()
        and config.meta_path.is_file()
    ):
        return False
    try:
        meta = json.loads(config.meta_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return False
    if not isinstance(meta, dict) or meta.get("tokenizer") != config.tokenizer:
        return False
    if config.tokenizer == "bpe":
        tok_file = meta.get("tokenizer_file")
        return (
            isinstance(tok_file, str)
            and meta.get("bpe_vocab_size") == config.bpe_vocab_size
            and (config.token_dir / tok_file).is_file()
        )
    return True


def prepare_tokens(config: TrainConfig, overwrite: bool = False) -> Path:
    """Download and tokenize dataset into config.token_dir unless already cached."""
    if not overwrite and tokens_ready(config):
        print(f"Using cached {config.tokenizer} tokens: {config.token_dir}")
        return config.token_dir

    ds = download(config)
    if config.tokenizer == "bpe":
        tokenized = tokenize_bpe(config, ds)
    elif config.tokenizer == "gpt2":
        tokenized = tokenize_gpt2(config, ds)
    else:
        tokenized = tokenize_char(config, ds)
    save_tensors(config, tokenized)
    return config.token_dir


def load_tokenizer(
    meta_path: Path | None = None,
) -> tuple[int, Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Return (vocab_size, encode, decode) from meta.json or default GPT-2 BPE."""
    if meta_path is not None and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tok_kind = meta.get("tokenizer")
        if tok_kind == "bpe":
            tok_file = meta_path.parent / meta.get(
                "tokenizer_file", _BPE_TOKENIZER_FILE
            )
            tok = Tokenizer.from_file(str(tok_file))

            def encode_bpe(text: str) -> list[int]:
                return tok.encode(text).ids

            def decode_bpe(ids: list[int]) -> str:
                return tok.decode(ids, skip_special_tokens=False)

            return tok.get_vocab_size(), encode_bpe, decode_bpe

        if tok_kind == "char":
            chars = meta["chars"]
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}

            def encode_char(text: str) -> list[int]:
                unknown = sorted({c for c in text if c not in stoi})
                if unknown:
                    raise ValueError(
                        f"Unsupported character(s) for char tokenizer: {unknown}"
                    )
                return [stoi[c] for c in text]

            def decode_char(ids: list[int]) -> str:
                return "".join(itos[i] for i in ids)

            return len(chars), encode_char, decode_char

        vocab_size = int(meta.get("vocab_size", _DEFAULT_GPT2_VOCAB_SIZE))
    else:
        vocab_size = _DEFAULT_GPT2_VOCAB_SIZE

    enc = _gpt2_encoding()

    def encode_gpt2(text: str) -> list[int]:
        return enc.encode(text, allowed_special={_EOT_TOKEN})

    return vocab_size, encode_gpt2, enc.decode
