
# Chat Shakespeare

A Shakespeare text generator implemented with a LLM transformer architecture using PyTorch.

## Description

- Built from the ground up from the classic paper that started it all: Attention is All You Need

## Getting Started

### Dependencies

* git clone
* uv sync

### Executing program

1. Run the main.py program
```
uv run main.py OR
marimo edit main.py
```

2. Adjust hyperparameters depending on hardware
```
device = cpu
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

device = gpu
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

This project were inspired by the Andrej Karpathy's Zero to Hero YouTube series as well as his nanoGPT repo.