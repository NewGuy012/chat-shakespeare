from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hyperparameters import TrainConfig
from model import GPT, GPTConfig
from sample import SampleConfig, sample
from tokenizer import load_tokenizer, save_tensors, tokenize_char, tokenize_gpt2
from train import get_lr, save_checkpoint, train


class PipelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        AcceleratorState._reset_state()
        PartialState._reset_state()

    def test_char_and_gpt2_tokenizers_use_json_meta(self) -> None:
        ds = {"train": {"text": ["hello world\n"]}, "test": {"text": ["hello!\n"]}}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig.auto("cpu", data_path=Path(tmp))
            save_tensors(cfg, tokenize_char(cfg, ds))
            self.assertTrue(cfg.meta_path.exists())
            self.assertFalse((cfg.data_path / "meta.pkl").exists())

            vocab_size, encode, decode = load_tokenizer(cfg.meta_path)
            self.assertEqual(vocab_size, cfg.vocab_size)
            self.assertEqual(decode(encode("hello!")), "hello!")
            with self.assertRaises(ValueError):
                encode("?")

            gpt2_tok = tokenize_gpt2(cfg, ds)
            self.assertGreater(len(gpt2_tok["train"]), 0)
            _, enc_gpt2, dec_gpt2 = load_tokenizer(cfg.meta_path)
            self.assertEqual(dec_gpt2(enc_gpt2("hello world")), "hello world")

    def test_presets_and_lr_schedule(self) -> None:
        cpu = TrainConfig.auto("cpu", max_iters=200)
        self.assertEqual(cpu.mixed_precision, "no")
        self.assertEqual(cpu.lr_decay_iters, 200)
        self.assertEqual(TrainConfig.auto("cuda").compile, True)
        self.assertEqual(TrainConfig.auto("mps").device, "mps")

        self.assertLess(get_lr(cpu, 0), cpu.learning_rate)
        self.assertEqual(get_lr(cpu, cpu.lr_decay_iters), cpu.min_lr)
        self.assertEqual(get_lr(TrainConfig.auto("cpu", lr_decay_iters=0, warmup_iters=0), 5), cpu.learning_rate)
        self.assertEqual(get_lr(TrainConfig.auto("cpu", lr_decay_iters=10, warmup_iters=10), 10), cpu.min_lr)

    def test_model_guards_and_from_pretrained_safetensors(self) -> None:
        model = GPT(GPTConfig(block_size=8, vocab_size=16, n_layer=1, n_head=2, n_embd=8))
        idx = torch.randint(0, 16, (1, 4))
        out = model.generate(idx, max_new_tokens=3, temperature=1.0, top_k=4)
        self.assertEqual(out.shape, (1, 7))
        with self.assertRaises(ValueError):
            model.generate(idx, max_new_tokens=1, temperature=0.0)
        with self.assertRaises(ValueError):
            model(torch.randint(0, 16, (1, 9)))

        tiny_spec = {"gpt2": {"n_layer": 1, "n_head": 2, "n_embd": 8}}
        with tempfile.TemporaryDirectory() as tmp, patch("model._PRETRAINED_CONFIGS", tiny_spec):
            raw_sd = {
                "wte.weight": torch.randn(50257, 8),
                "wpe.weight": torch.randn(1024, 8),
                "h.0.ln_1.weight": torch.ones(8),
                "h.0.ln_1.bias": torch.zeros(8),
                "h.0.attn.bias": torch.tril(torch.ones(1, 1, 1024, 1024)),
                "h.0.attn.c_attn.weight": torch.randn(8, 24),
                "h.0.attn.c_attn.bias": torch.zeros(24),
                "h.0.attn.c_proj.weight": torch.randn(8, 8),
                "h.0.attn.c_proj.bias": torch.zeros(8),
                "h.0.ln_2.weight": torch.ones(8),
                "h.0.ln_2.bias": torch.zeros(8),
                "h.0.mlp.c_fc.weight": torch.randn(8, 32),
                "h.0.mlp.c_fc.bias": torch.zeros(32),
                "h.0.mlp.c_proj.weight": torch.randn(32, 8),
                "h.0.mlp.c_proj.bias": torch.zeros(8),
                "ln_f.weight": torch.ones(8),
                "ln_f.bias": torch.zeros(8),
            }
            weights_path = Path(tmp) / "model.safetensors"
            save_file(raw_sd, str(weights_path))
            loaded = GPT.from_pretrained("gpt2", weights_path=weights_path)
            self.assertTrue(torch.equal(loaded.transformer.h[0].attn.c_attn.weight, raw_sd["h.0.attn.c_attn.weight"].t()))
            self.assertIs(loaded.transformer.wte.weight, loaded.lm_head.weight)
            loaded.crop_block_size(32)
            self.assertEqual(loaded.config.block_size, 32)

    def test_accelerate_train_checkpoint_and_sample_e2e(self) -> None:
        ds = {
            "train": {"text": ["First Citizen:\nBefore we proceed any further, hear me speak.\n" * 8]},
            "test": {"text": ["All:\nSpeak, speak.\n" * 6]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig.auto(
                "cpu",
                data_path=Path(tmp),
                block_size=8,
                batch_size=2,
                gradient_accumulation_steps=2,
                n_layer=1,
                n_head=2,
                n_embd=16,
                max_iters=4,
                warmup_iters=1,
                log_iters=2,
                eval_iters=2,
            )
            save_tensors(cfg, tokenize_char(cfg, ds))
            acc = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
            _, trained_model, losses = train(cfg, accelerator=acc)
            self.assertIn("train", losses)
            self.assertIn("val", losses)

            ckpt_dir = save_checkpoint(cfg, acc, trained_model, losses)
            self.assertTrue((ckpt_dir / "model.safetensors").exists())
            self.assertTrue((ckpt_dir / "config.json").exists())

            texts = sample(cfg, SampleConfig(start="First", num_samples=1, max_new_tokens=6, top_k=5))
            self.assertEqual(len(texts), 1)
            self.assertTrue(texts[0].startswith("First"))


if __name__ == "__main__":
    unittest.main()
