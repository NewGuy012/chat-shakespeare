import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import torch
    import pickle
    import tiktoken
    import marimo as mo
    from dataclasses import dataclass


@app.class_definition
@dataclass
class SampleConfig:
    load_meta: bool = False
    start: str = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 1 # number of samples to draw
    max_new_tokens: int = 500 # number of tokens generated in each sample
    temperature: float = 0.95 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 200


@app.function
def sample(config, sample_config, model):
    root_path = config["root_path"]
    device  = config["device"]
    compile = config["compile"]

    load_meta = sample_config.load_meta
    start = sample_config.start
    num_samples = sample_config.num_samples
    max_new_tokens = sample_config.max_new_tokens
    temperature = sample_config.temperature
    top_k = sample_config.top_k

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    if load_meta:
        meta_path = root_path / "data" / "meta.pkl"
        
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
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
