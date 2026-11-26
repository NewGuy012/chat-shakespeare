from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from hyperparameters import (
    DEFAULT_BPE_VOCAB_SIZE,
    TrainConfig,
    is_int,
    read_json,
    read_json_dict,
    read_safetensors_keys,
    write_json_atomic,
)

_DATASET_REPO = "Trelis/tiny-shakespeare"
_DATASET_REVISION = "2225a5674b252d623cfb151ef099d179629c2f49"
_EOT_TOKEN = "<|endoftext|>"
_BPE_TOKENIZER_FILE = "tokenizer.json"


@lru_cache(maxsize=1)
def _gpt2_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def _split_texts(ds: DatasetDict | dict[str, Any]) -> tuple[list[str], list[str]]:
    val_split = "validation" if "validation" in ds else "test"
    train_items = list(ds["train"]["text"])
    val_items = list(ds[val_split]["text"])
    if not any(train_items):
        raise ValueError("Cannot tokenize empty training dataset")
    return train_items, val_items


def _save_safetensors_atomic(tensors: dict[str, torch.Tensor], path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        save_file(tensors, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def download(config: TrainConfig) -> DatasetDict:
    """Download or load cached tiny-shakespeare dataset."""
    if config.download_path.exists():
        loaded = load_from_disk(config.download_path)
        if not isinstance(loaded, DatasetDict):
            raise TypeError(
                f"Expected DatasetDict at {config.download_path}, got {type(loaded).__name__}"
            )
        return loaded

    dataset = load_dataset(_DATASET_REPO, revision=_DATASET_REVISION)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict from load_dataset, got {type(dataset).__name__}"
        )
    if "Text" in dataset["train"].column_names:
        dataset = dataset.rename_column("Text", "text")

    config.download_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=config.download_path.parent, prefix=".dl."))
    try:
        dataset.save_to_disk(tmp_dir)
        os.replace(tmp_dir, config.download_path)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return dataset


def tokenize_char(
    ds: DatasetDict | dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Character-level tokenize train/val splits and return (tokenized, meta)."""
    train_items, val_items = _split_texts(ds)
    train_text = "".join(train_items)
    val_text = "".join(val_items)
    chars = sorted(set(train_text + val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    print(f"\nTokenize text (char): vocab_size={len(chars):,}")
    return (
        {"train": [stoi[c] for c in train_text], "val": [stoi[c] for c in val_text]},
        {"tokenizer": "char", "vocab_size": len(chars), "chars": "".join(chars)},
    )


def tokenize_bpe(
    ds: DatasetDict | dict[str, Any],
    vocab_size: int = DEFAULT_BPE_VOCAB_SIZE,
) -> tuple[dict[str, list[int]], dict[str, Any], Tokenizer]:
    """Train a byte-level BPE tokenizer and return (tokenized, meta, tokenizer)."""
    train_items, val_items = _split_texts(ds)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
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
    print(
        f"\nTokenize text (bpe): vocab_size={actual_vocab_size:,} (target={vocab_size:,})"
    )
    return (
        {"train": encode_split(train_items), "val": encode_split(val_items)},
        {
            "tokenizer": "bpe",
            "vocab_size": actual_vocab_size,
            "bpe_vocab_size": vocab_size,
        },
        tok,
    )


def tokenize_gpt2(
    ds: DatasetDict | dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Tokenize train/val splits with GPT-2 BPE (tiktoken) and return (tokenized, meta)."""
    enc = _gpt2_encoding()
    train_items, val_items = _split_texts(ds)

    def encode_split(items: list[str]) -> list[int]:
        ids: list[int] = []
        for text in items:
            ids.extend(enc.encode_ordinary(text))
            ids.append(enc.eot_token)
        return ids

    print(f"\nTokenize text (gpt2): vocab_size={enc.n_vocab:,}")
    return (
        {"train": encode_split(train_items), "val": encode_split(val_items)},
        {"tokenizer": "gpt2", "vocab_size": enc.n_vocab},
    )


def save_tokens(
    config: TrainConfig,
    tokenized: dict[str, list[int]],
    meta: dict[str, Any],
    bpe_tokenizer: Tokenizer | None = None,
) -> None:
    """Persist tokenized splits, optional BPE tokenizer, and meta.json (written last)."""
    config.token_dir.mkdir(parents=True, exist_ok=True)
    for split, path in (("train", config.train_path), ("val", config.val_path)):
        _save_safetensors_atomic(
            {split: torch.tensor(tokenized[split], dtype=torch.long)}, path
        )
    if bpe_tokenizer is not None:
        tok_path = config.token_dir / _BPE_TOKENIZER_FILE
        fd, tmp_name = tempfile.mkstemp(
            dir=config.token_dir, prefix=f".{_BPE_TOKENIZER_FILE}.", suffix=".tmp"
        )
        os.close(fd)
        try:
            bpe_tokenizer.save(tmp_name)
            os.replace(tmp_name, tok_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    write_json_atomic(config.meta_path, meta)


def tokens_ready(config: TrainConfig) -> bool:
    """Return True when valid tokenized splits and matching meta.json exist on disk."""
    meta = read_json(config.meta_path)
    if meta is None or meta.get("tokenizer") != config.tokenizer:
        return False
    vocab_size = meta.get("vocab_size")
    if not is_int(vocab_size) or vocab_size < 1:
        return False
    train_keys = read_safetensors_keys(config.train_path)
    val_keys = read_safetensors_keys(config.val_path)
    if train_keys is None or "train" not in train_keys:
        return False
    if val_keys is None or "val" not in val_keys:
        return False
    if config.tokenizer == "bpe":
        return (
            meta.get("bpe_vocab_size") == config.bpe_vocab_size
            and (config.token_dir / _BPE_TOKENIZER_FILE).is_file()
        )
    if config.tokenizer == "char":
        chars = meta.get("chars")
        return isinstance(chars, str) and len(chars) > 0
    return True


def prepare_tokens(config: TrainConfig, overwrite: bool = False) -> Path:
    """Download and tokenize dataset into config.token_dir unless already cached."""
    if not overwrite and tokens_ready(config):
        print(f"Using cached {config.tokenizer} tokens: {config.token_dir}")
        return config.token_dir

    ds = download(config)
    bpe_tok: Tokenizer | None = None
    if config.tokenizer == "bpe":
        tokenized, meta, bpe_tok = tokenize_bpe(ds, config.bpe_vocab_size)
    elif config.tokenizer == "gpt2":
        tokenized, meta = tokenize_gpt2(ds)
    else:
        tokenized, meta = tokenize_char(ds)
    save_tokens(config, tokenized, meta, bpe_tok)
    return config.token_dir


def load_tokenizer(
    meta_path: Path | None = None,
) -> tuple[int, Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Return (vocab_size, encode, decode) from meta.json or GPT-2 BPE."""
    if meta_path is not None:
        meta = read_json_dict(meta_path)
        tok_kind = meta.get("tokenizer")
        if tok_kind == "bpe":
            # Fixed filename prevents untrusted meta.json from traversing outside token_dir.
            tok = Tokenizer.from_file(str(meta_path.parent / _BPE_TOKENIZER_FILE))

            def encode_bpe(text: str) -> list[int]:
                return tok.encode(text).ids

            def decode_bpe(ids: list[int]) -> str:
                return tok.decode(ids, skip_special_tokens=False)

            return tok.get_vocab_size(), encode_bpe, decode_bpe

        if tok_kind == "char":
            chars = meta.get("chars")
            if not isinstance(chars, str) or not chars:
                raise ValueError(f"Missing or invalid 'chars' in {meta_path}")
            stoi = {c: i for i, c in enumerate(chars)}
            itos = dict(enumerate(chars))

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

        if tok_kind != "gpt2":
            raise ValueError(f"Unsupported tokenizer {tok_kind!r} in {meta_path}")

    enc = _gpt2_encoding()

    def encode_gpt2(text: str) -> list[int]:
        return enc.encode(text, allowed_special={_EOT_TOKEN})

    return enc.n_vocab, encode_gpt2, enc.decode
