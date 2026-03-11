import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import torch
    import pickle
    import tiktoken
    import marimo as mo

    from pathlib import Path, WindowsPath, PosixPath
    from dataclasses import dataclass
    from train import initialize_model
    from hyperparameters import determine_device

    torch.set_float32_matmul_precision('high')


@app.class_definition
@dataclass
class SampleConfig:
    start: str = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 1 # number of samples to draw
    max_new_tokens: int = 500 # number of tokens generated in each sample
    temperature: float = 0.95 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 200
    bOverwrite: bool = False


@app.function
def load_checkpoint(checkpoint_path):
    device, _ = determine_device()

    torch.serialization.add_safe_globals([WindowsPath, PosixPath])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    model, optimizer = initialize_model(config)

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    return config, model, optimizer


@app.function
def sample(config, sample_config):
    checkpoint_path = config["checkpoint_path"]
    config, model, _ = load_checkpoint(checkpoint_path)

    meta_path = config["meta_path"]
    device  = config["device"]
    compile = config["compile"]

    start = sample_config.start
    num_samples = sample_config.num_samples
    max_new_tokens = sample_config.max_new_tokens
    temperature = sample_config.temperature
    top_k = sample_config.top_k

    model.eval()
    model.to(device)

    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        encode = meta['encode']
        decode = meta['decode']

        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # gpt-2 encodings by default
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')


if __name__ == "__main__":
    app.run()
