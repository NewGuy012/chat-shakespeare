from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState
from safetensors.torch import save_file, save_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hyperparameters import GPT2_VOCAB_SIZE, TrainConfig
from model import GPT, Block, GPTConfig, KVCache, RMSNorm
from sample import SampleConfig, sample
from tokenizer import (
    load_tokenizer,
    prepare_tokens,
    save_tokens,
    tokenize_bpe,
    tokenize_char,
    tokenize_gpt2,
    tokens_ready,
)
from train import (
    TokenBatcher,
    build_accelerator,
    checkpoint_ready,
    ensure_checkpoint,
    get_lr,
    initialize_model,
    load_checkpoint,
    save_checkpoint,
    train,
)
from train import (
    main as train_main,
)

_DS = {
    "train": {
        "text": ["First Citizen:\nBefore we proceed any further, hear me speak.\n" * 16]
    },
    "test": {"text": ["All:\nSpeak, speak.\n" * 20]},
}


def _tiny_cfg(tmp: str, **overrides: object) -> TrainConfig:
    defaults: dict[str, object] = {
        "data_path": Path(tmp),
        "block_size": 8,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "max_iters": 2,
        "warmup_iters": 1,
        "log_iters": 1,
        "eval_iters": 1,
    }
    defaults.update(overrides)
    return TrainConfig.auto("cpu", **defaults)


def _reset_acc() -> None:
    AcceleratorState._reset_state()
    PartialState._reset_state()


class PipelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        _reset_acc()

    def test_tokenizers_prepare_and_security(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bpe_cfg = _tiny_cfg(tmp)
            bpe_512 = _tiny_cfg(tmp, vocab_size=512)
            char_cfg = _tiny_cfg(tmp, tokenizer="char")
            gpt2_cfg = _tiny_cfg(tmp, init_from="gpt2")

            tok_bpe, meta_bpe, obj_bpe = tokenize_bpe(_DS, bpe_cfg.bpe_vocab_size)
            save_tokens(bpe_cfg, tok_bpe, meta_bpe, obj_bpe)
            vocab, enc, dec = load_tokenizer(bpe_cfg.meta_path)
            self.assertEqual(vocab, meta_bpe["vocab_size"])
            self.assertEqual(
                dec(enc("hello? — <|endoftext|>")), "hello? — <|endoftext|>"
            )

            # Path traversal in meta.json cannot escape token_dir.
            poisoned = {**meta_bpe, "tokenizer_file": "../../etc/passwd"}
            bpe_cfg.meta_path.write_text(json.dumps(poisoned), encoding="utf-8")
            self.assertEqual(load_tokenizer(bpe_cfg.meta_path)[0], vocab)

            tok_char, meta_char = tokenize_char(_DS)
            save_tokens(char_cfg, tok_char, meta_char)
            v_c, enc_c, dec_c = load_tokenizer(char_cfg.meta_path)
            self.assertEqual(v_c, meta_char["vocab_size"])
            self.assertEqual(dec_c(enc_c("First:")), "First:")
            with self.assertRaises(ValueError):
                enc_c("?")

            tok_gpt2, meta_gpt2 = tokenize_gpt2(_DS)
            self.assertEqual(meta_gpt2["vocab_size"], GPT2_VOCAB_SIZE)
            self.assertGreater(len(tok_gpt2["train"]), 0)
            self.assertEqual(load_tokenizer(None)[0], GPT2_VOCAB_SIZE)

            # Cache hits, isolation, overwrite, and corruption checks.
            with patch("tokenizer.download", return_value=_DS) as mock_dl:
                prepare_tokens(bpe_cfg)
                self.assertEqual(mock_dl.call_count, 0)
                prepare_tokens(bpe_512)
                prepare_tokens(gpt2_cfg)
                self.assertEqual(mock_dl.call_count, 2)
                self.assertTrue(tokens_ready(bpe_512))
                self.assertTrue(tokens_ready(gpt2_cfg))

                (bpe_cfg.token_dir / "tokenizer.json").unlink()
                self.assertFalse(tokens_ready(bpe_cfg))
                prepare_tokens(bpe_cfg)
                self.assertEqual(mock_dl.call_count, 3)

                bpe_cfg.train_path.write_bytes(b"corrupt")
                self.assertFalse(tokens_ready(bpe_cfg))
                prepare_tokens(bpe_cfg)
                self.assertEqual(mock_dl.call_count, 4)

                prepare_tokens(char_cfg, overwrite=True)
                self.assertEqual(mock_dl.call_count, 5)

            char_cfg.meta_path.write_text('{"tokenizer": "evil"}', encoding="utf-8")
            self.assertFalse(tokens_ready(char_cfg))
            with self.assertRaises(ValueError):
                load_tokenizer(char_cfg.meta_path)

    def test_config_presets_validation_and_fingerprint(self) -> None:
        cpu = TrainConfig.auto("cpu", max_iters=200, learning_rate=5e-4)
        self.assertEqual(
            (cpu.arch, cpu.mixed_precision, cpu.compile_mode),
            ("modern", "no", "default"),
        )
        self.assertEqual(cpu.lr_decay_iters, 200)
        self.assertAlmostEqual(cpu.min_lr, 5e-5)
        self.assertEqual(TrainConfig.auto("mps").device, "mps")

        for vram_gb, exp_bs, exp_acc, exp_iters, exp_b2 in (
            (8, 32, 2, 5000, 0.99),
            (16, 64, 1, 5000, 0.99),
            (48, 128, 1, 2500, 0.95),
        ):
            props = SimpleNamespace(total_memory=vram_gb * 1024**3)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.current_device", return_value=0),
                patch("torch.cuda.get_device_properties", return_value=props),
                patch("torch.cuda.is_bf16_supported", return_value=True),
            ):
                cuda_cfg = TrainConfig.auto("cuda")
                self.assertEqual(
                    (
                        cuda_cfg.batch_size,
                        cuda_cfg.gradient_accumulation_steps,
                        cuda_cfg.max_iters,
                        cuda_cfg.lr_decay_iters,
                        cuda_cfg.beta2,
                        cuda_cfg.mixed_precision,
                        cuda_cfg.compile_mode,
                    ),
                    (
                        exp_bs,
                        exp_acc,
                        exp_iters,
                        exp_iters,
                        exp_b2,
                        "bf16",
                        "reduce-overhead",
                    ),
                )

        self.assertLess(get_lr(cpu, 0), cpu.learning_rate)
        self.assertEqual(get_lr(cpu, cpu.lr_decay_iters), cpu.min_lr)
        self.assertEqual(
            get_lr(TrainConfig.auto("cpu", lr_decay_iters=0, warmup_iters=0), 5),
            TrainConfig.auto("cpu").learning_rate,
        )

        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            a = TrainConfig.auto("cpu", data_path=Path(t1))
            b = TrainConfig.auto(
                "cpu", data_path=Path(t2), log_iters=5, eval_iters=2, compile=True
            )
            self.assertEqual(a.fingerprint(), b.fingerprint())
            self.assertEqual(a.checkpoint_name, b.checkpoint_name)

            before = a.checkpoint_dir
            a.vocab_size = 300
            self.assertEqual(a.checkpoint_dir, before)
            self.assertNotEqual(
                TrainConfig.auto("cpu", data_path=Path(t1), arch="gpt2").checkpoint_dir,
                before,
            )
            self.assertNotEqual(
                TrainConfig.auto(
                    "cpu", data_path=Path(t1), vocab_size=512
                ).checkpoint_dir,
                before,
            )

            gpt2_direct = TrainConfig(init_from="gpt2")
            self.assertEqual(
                (gpt2_direct.arch, gpt2_direct.tokenizer), ("gpt2", "gpt2")
            )

        bad_cases: tuple[dict[str, Any], ...] = (
            {"arch": "unknown"},
            {"compile_mode": "turbo"},
            {"tokenizer": "unknown"},
            {"bpe_vocab_size": 100},
            {"n_embd": 12, "n_head": 4, "arch": "modern"},
            {"device": "mps", "mixed_precision": "fp16"},
            {"batch_size": True},
            {"learning_rate": 1e-4, "min_lr": 1e-3},
            {"weight_decay": -0.1},
        )
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                TrainConfig(**bad)
        with self.assertRaises(ValueError):
            TrainConfig.auto("cpu", vocab_size=100)

    def test_model_kv_cache_generate_and_pretrained(self) -> None:
        for arch in ("modern", "gpt2"):
            model = GPT(
                GPTConfig(
                    block_size=8,
                    vocab_size=16,
                    n_layer=1,
                    n_head=2,
                    n_embd=8,
                    arch=arch,
                )
            )
            model.eval()
            block0 = model.transformer.h[0]
            assert isinstance(block0, Block)
            self.assertIsInstance(
                block0.ln_1, RMSNorm if arch == "modern" else torch.nn.LayerNorm
            )

            idx = torch.randint(0, 16, (1, 4))
            logits_none, loss = model(
                idx, torch.randint(0, 16, (1, 4)), return_logits=False
            )
            self.assertIsNone(logits_none)
            assert loss is not None

            # KV cache prefill + decode matches uncached forward logits.
            full_idx = torch.randint(0, 16, (2, 6))
            uncached, _ = model(full_idx)
            cache = KVCache(1, 8)
            model(full_idx[:, :4], pos_offset=0, kv_cache=cache)
            model(full_idx[:, 4:5], pos_offset=4, kv_cache=cache)
            cached, _ = model(full_idx[:, 5:6], pos_offset=5, kv_cache=cache)
            assert uncached is not None and cached is not None
            self.assertTrue(torch.allclose(uncached, cached, atol=1e-4))

            out = model.generate(idx, max_new_tokens=9, top_k=4, valid_vocab_size=10)
            self.assertEqual(out.shape, (1, 13))
            self.assertTrue(bool((out[:, 4:] < 10).all()))

            model.crop_block_size(4)
            self.assertEqual(model.config.block_size, 4)
            if model.transformer.wpe is not None:
                self.assertEqual(model.transformer.wpe.num_embeddings, 4)

            with self.assertRaises(ValueError):
                model.generate(idx, max_new_tokens=1, temperature=0.0)
            with self.assertRaises(ValueError):
                model.generate(idx, max_new_tokens=1, top_k=0)
            with self.assertRaises(ValueError):
                model(idx, pos_offset=-1)
            with self.assertRaises(ValueError):
                model(torch.randint(0, 16, (1, 5)))

        tiny_spec = {"gpt2": {"n_layer": 1, "n_head": 2, "n_embd": 8}}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("model.PRETRAINED_CONFIGS", tiny_spec),
        ):
            ref = GPT(
                GPTConfig(
                    block_size=1024,
                    vocab_size=GPT2_VOCAB_SIZE,
                    n_layer=1,
                    n_head=2,
                    n_embd=8,
                    bias=True,
                    arch="gpt2",
                )
            )
            conv_keys = (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")
            raw_sd = {
                k.removeprefix("transformer."): (
                    v.t().contiguous() if k.endswith(conv_keys) else v.clone()
                )
                for k, v in ref.state_dict().items()
                if k != "lm_head.weight"
            }
            raw_sd["h.0.attn.bias"] = torch.tril(torch.ones(1, 1, 1024, 1024))
            weights_path = Path(tmp) / "model.safetensors"
            save_file(raw_sd, str(weights_path))

            loaded = GPT.from_pretrained("gpt2", weights_path=weights_path)
            self.assertEqual(
                (loaded.config.arch, loaded.config.vocab_size), ("gpt2", 50304)
            )
            self.assertTrue(
                torch.equal(
                    loaded.transformer.wte.weight[:GPT2_VOCAB_SIZE],
                    raw_sd["wte.weight"],
                )
            )
            self.assertTrue(
                bool((loaded.transformer.wte.weight[GPT2_VOCAB_SIZE:] == 0).all())
            )
            self.assertIs(loaded.transformer.wte.weight, loaded.lm_head.weight)

    def test_train_checkpoint_portability_and_sample(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tokenizer.download", return_value=_DS),
        ):
            cfg = _tiny_cfg(tmp, max_iters=4, log_iters=2, eval_iters=2)
            prepare_tokens(cfg)

            ckpt_dir = ensure_checkpoint(cfg)
            self.assertEqual(ckpt_dir, cfg.checkpoint_dir)
            self.assertTrue(checkpoint_ready(cfg))

            # Portable config.json omits host data_path; load_checkpoint infers it from checkpoints/.
            saved_cfg = json.loads(
                (ckpt_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("data_path", saved_cfg)

            ckpt_cfg, loaded_model = load_checkpoint(ckpt_dir, device="cpu")
            self.assertEqual(ckpt_cfg.checkpoint_dir, ckpt_dir)
            self.assertEqual(ckpt_cfg.vocab_size, loaded_model.config.vocab_size)

            # Non-training fields hit cache; training fields isolate; overwrite retrains.
            same_cfg = _tiny_cfg(tmp, max_iters=4, log_iters=99, eval_iters=2)
            with patch("train.train") as mock_train:
                self.assertEqual(ensure_checkpoint(same_cfg), ckpt_dir)
                mock_train.assert_not_called()

            _reset_acc()
            other_dir = ensure_checkpoint(_tiny_cfg(tmp, max_iters=3))
            self.assertNotEqual(other_dir, ckpt_dir)

            # Corrupt weights invalidate checkpoint_ready.
            (ckpt_dir / "model.safetensors").write_bytes(b"bad")
            self.assertFalse(checkpoint_ready(cfg))
            _reset_acc()
            ensure_checkpoint(cfg)
            self.assertTrue(checkpoint_ready(cfg))

            # Vocab padding masks invalid token IDs during batched sampling.
            char_cfg = _tiny_cfg(tmp, tokenizer="char")
            prepare_tokens(char_cfg)
            raw_vocab = load_tokenizer(char_cfg.meta_path)[0]
            char_model, _ = initialize_model(char_cfg)
            self.assertEqual(char_model.config.vocab_size, 64)
            with torch.no_grad():
                char_model.transformer.wte.weight[raw_vocab:].fill_(50.0)
            texts = sample(
                char_cfg,
                SampleConfig(start="First", num_samples=3, max_new_tokens=6, top_k=5),
                model=char_model,
            )
            self.assertEqual(len(texts), 3)
            self.assertTrue(all(t.startswith("First") and len(t) == 11 for t in texts))

            disk_texts = sample(
                cfg,
                SampleConfig(start="First", num_samples=1, max_new_tokens=4, top_k=5),
            )
            self.assertEqual(len(disk_texts), 1)
            self.assertTrue(disk_texts[0].startswith("First"))

            # CPU-backed TokenBatcher fallback path (simulating >256MB split).
            cpu_batcher = TokenBatcher(
                torch.arange(40),
                batch_size=2,
                block_size=8,
                target_device=torch.device("cpu"),
            )
            with patch("train._MAX_DEVICE_TOKENS_BYTES", 1):
                _reset_acc()
                _, _, cpu_losses = train(_tiny_cfg(tmp, max_iters=2))
                self.assertIn("val", cpu_losses)
            bx, by = cpu_batcher.next_batch()
            self.assertEqual((bx.shape, by.shape), ((2, 8), (2, 8)))

            # Early stopping restores best val checkpoint.
            _reset_acc()
            es_cfg = _tiny_cfg(
                tmp, max_iters=10, log_iters=2, early_stopping_patience=1
            )
            eval_seq = iter(
                [
                    {"train": 5.0, "val": 5.0},
                    {"train": 4.5, "val": 4.0},
                    {"train": 4.2, "val": 4.3},
                ]
            )
            with patch(
                "train.estimate_loss", side_effect=lambda *a, **k: next(eval_seq)
            ):
                _, _, es_losses = train(
                    es_cfg, accelerator=Accelerator(cpu=True, mixed_precision="no")
                )
            self.assertEqual(es_losses["val"], 4.0)

        with self.assertRaises(ValueError):
            SampleConfig(top_k=0)

    def test_legacy_gpt2_detection_and_cli(self) -> None:
        self.assertEqual(
            TrainConfig.from_dict({"init_from": "scratch", "vocab_size": 65}).tokenizer,
            "char",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tokenizer.download", return_value=_DS),
        ):
            ckpt_dir = Path(tmp) / "legacy_gpt2"
            ckpt_dir.mkdir()
            save_model(
                GPT(
                    GPTConfig(
                        block_size=8,
                        vocab_size=64,
                        n_layer=1,
                        n_head=2,
                        n_embd=8,
                        bias=False,
                        arch="gpt2",
                    )
                ),
                str(ckpt_dir / "model.safetensors"),
            )
            legacy_cfg = {
                "block_size": 8,
                "vocab_size": 64,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 8,
                "bias": False,
                "init_from": "scratch",
                "tokenizer": "char",
                "arch": "modern",
            }
            (ckpt_dir / "config.json").write_text(
                json.dumps(legacy_cfg), encoding="utf-8"
            )

            auto_cfg = TrainConfig.auto(
                "cpu", data_path=Path(tmp), init_from=str(ckpt_dir)
            )
            self.assertEqual(auto_cfg.arch, "gpt2")
            self.assertEqual(load_checkpoint(ckpt_dir)[1].config.arch, "gpt2")
            self.assertEqual(initialize_model(auto_cfg)[0].config.arch, "gpt2")

            cuda_acc = build_accelerator(TrainConfig.auto("cuda", data_path=Path(tmp)))
            assert cuda_acc.state.dynamo_plugin is not None
            self.assertEqual(cuda_acc.state.dynamo_plugin.mode, "reduce-overhead")

            cases = (
                ([], "modern", "bpe", 1024),
                (["--vocab-size", "512"], "modern", "bpe", 512),
                (["--tokenizer", "char"], "modern", "char", 1024),
                (["--init-from", "gpt2"], "gpt2", "gpt2", 50304),
                (
                    [
                        "--arch",
                        "gpt2",
                        "--batch-size",
                        "4",
                        "--grad-accum",
                        "1",
                        "--block-size",
                        "32",
                        "--lr",
                        "5e-4",
                        "--no-compile",
                        "--early-stopping",
                        "2",
                    ],
                    "gpt2",
                    "bpe",
                    1024,
                ),
            )
            for extra_args, exp_arch, exp_tok, exp_vocab in cases:
                _reset_acc()
                argv = [
                    "src/train.py",
                    "--device",
                    "cpu",
                    "--data-path",
                    tmp,
                    "--max-iters",
                    "2",
                    *extra_args,
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch("train.ensure_checkpoint") as mock_ensure,
                ):
                    train_main()
                    called = mock_ensure.call_args.args[0]
                    self.assertEqual(
                        (called.arch, called.tokenizer, called.vocab_size),
                        (exp_arch, exp_tok, exp_vocab),
                    )

            with (
                patch.object(
                    sys,
                    "argv",
                    ["src/train.py", "--init-from", str(Path(tmp) / "missing")],
                ),
                self.assertRaises(SystemExit),
            ):
                train_main()

    @unittest.skipUnless(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "MPS device not available",
    )
    def test_mps_bf16_modern_e2e(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tokenizer.download", return_value=_DS),
        ):
            cfg = TrainConfig.auto(
                "mps",
                data_path=Path(tmp),
                block_size=16,
                batch_size=4,
                gradient_accumulation_steps=1,
                n_layer=1,
                n_head=2,
                n_embd=32,
                max_iters=3,
                warmup_iters=1,
                log_iters=2,
                eval_iters=1,
            )
            prepare_tokens(cfg)
            acc, trained_model, losses = train(cfg)
            self.assertIn("val", losses)
            ckpt_dir = save_checkpoint(cfg, acc, trained_model, losses)
            self.assertTrue((ckpt_dir / "model.safetensors").exists())
            texts = sample(
                cfg,
                SampleConfig(start="First", num_samples=2, max_new_tokens=8, top_k=5),
            )
            self.assertEqual(len(texts), 2)
            self.assertTrue(all(t.startswith("First") for t in texts))


if __name__ == "__main__":
    unittest.main()
