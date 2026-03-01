
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

0. Set up dependencies
```
git clone
uv sync
```

1. Run the main.py program
```
uv run main.py OR
marimo edit main.py
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
```

3. For using GPU compute on cloud, I'd recommend Thunder Compute

## Authors

Moses Yoo, juyoung.m.yoo at gmail dot com.

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the BSD 1-Clause license. See the LICENSE.md file for details.

## Acknowledgments

This project were inspired by the Andrej Karpathy's Zero to Hero YouTube playlist as well as his nanoGPT repo.

## To Do
* huggingface accelerate
* change dataset to wiki
* fine-tune to conversational
* add support to thunder compute
* add sequential data loader
* add support for running inference on pre-trained models
* update readme in github