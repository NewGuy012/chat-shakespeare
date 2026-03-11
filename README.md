
# Chat Shakespeare

A sandbox to explore concepts in LLM and PyTorch.

## Description

- Built from the ground up from the classic LLM transformer paper that started it all: Attention is All You Need.
- Trained on the tinyshakespeare dataset.

## Getting Started

### Main Dependencies

* marimo
* torch (PyTorch)
* transformers (Hugging Face)

### Executing program

1. Setup using setup.sh
```
sudo ldconfig
git clone https://github.com/NewGuy012/chat-shakespeare.git
cd chat-shakespeare/
uv sync
source .venv/bin/activate
marimo edit
```

2. Adjust hyperparameters depending on hardware
```
device=cpu
-------------
compile=False
eval_iters=20
log_interval=1
block_size=64
batch_size=12
n_layer=4
n_head=4
n_embd=128
max_iters=2000
lr_decay_iters=2000
dropout=0.0

device=gpu
-------------
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
```

3. For using GPU compute on cloud, I am currently using Thunder Compute.

## Authors

Moses Yoo, juyoung.m.yoo at gmail dot com.

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the BSD 2-Clause license. See the LICENSE.md file for details.

## Acknowledgments

This project were inspired by the Andrej Karpathy's Zero to Hero YouTube playlist as well as his nanoGPT repo.

## To Do
* huggingface accelerate vs pytorch lightning
* change dataset to wiki
* fine-tune to conversational