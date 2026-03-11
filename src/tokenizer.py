import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import torch
    import pickle
    import tiktoken
    import marimo as mo

    from pathlib import Path
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from safetensors.torch import save_file
    from datasets import DatasetDict, load_dataset, load_from_disk

    enc = tiktoken.get_encoding("gpt2")


@app.function
def download(config):
    download_path = config["download_path"]
    
    if download_path.exists():
        dataset = load_from_disk(download_path)
    else:
        # jchwenger/tiny_shakespeare 
        dataset = load_dataset("Trelis/tiny-shakespeare")
        dataset = dataset.rename_column("Text", "text")
    
        dataset.save_to_disk(download_path)

    return dataset


@app.function
def tokenize_gpt2(config, ds):
    # Load the standard Python tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def tokenization(ds):
        token_dict = tokenizer(ds["text"])

        input_ids = token_dict["input_ids"]

        for input_id in input_ids:
            input_id.append(enc.eot_token)

        token_dict["input_ids"] = input_ids

        return token_dict

    # Batch tokenize
    tokenized = ds.map(
        tokenization,
        remove_columns=['text'],
        desc="HF tokenizer",
        batched=True)

    meta = {
        'vocab_size': config["vocab_size"],
        'encode': encode_gpt2,
        'decode': decode_gpt2,
    }
    save_meta(config, meta)

    return tokenized


@app.function
def encode_gpt2(s):
    return enc.encode(s, allowed_special={"<|endoftext|>"})


@app.function
def decode_gpt2(l):
    return enc.decode(l)


@app.function
def tokenize_tiktoken(config, ds):
    def process(ds):
        ids = enc.encode_ordinary(ds['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits"
    )

    meta = {
        'vocab_size': config["vocab_size"],
        'encode': encode_gpt2,
        'decode': decode_gpt2,
    }
    save_meta(config, meta)

    return tokenized


@app.function
def tokenize_char(config, ds):
    data_path = config["data_path"]
    tensor_path = data_path / "train.safetensors"

    ds_train = ds["train"]["text"]
    ds_val = ds["test"]["text"]

    ds_train = [item for sublist in ds_train for item in sublist]
    ds_val = [item for sublist in ds_val for item in sublist]
    data = ds_train + ds_val

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    unique_chars = ''.join(chars)

    print("\nTokenize text:")
    print(f"\tVocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # encode both to integers
    ds_train = encode_char(ds_train, stoi)
    ds_val = encode_char(ds_val, stoi)

    ds = {
        "train": {"input_ids": ds_train},
        "test": {"input_ids": ds_val}
    }

    meta = {
        'vocab_size': vocab_size,
        'encode': encode_char,
        'decode': decode_char,
        'stoi': stoi,
        'itos': itos
    }
    save_meta(config, meta)

    return ds


@app.function
def encode_char(s, stoi):
    return [stoi[c] for c in s]


@app.function
def decode_char(l, itos):
    return ''.join([itos[i] for i in l])


@app.function
def save_meta(config, meta):
    data_path = config["data_path"]

    save_path = data_path / "meta.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(meta, f)


@app.function
def save_tensors(config, ds):
    data_path = config["data_path"]
    train_path = data_path / "train.safetensors"
    val_path = data_path / "val.safetensors"

    if isinstance(ds, DatasetDict):
        ds.set_format(type='torch', columns=['input_ids'])

        ds_train = torch.cat(ds["train"]["input_ids"][:])
        ds_val = torch.cat(ds["test"]["input_ids"][:])

    else:
        ds_train = torch.tensor(ds["train"]["input_ids"])
        ds_val = torch.tensor(ds["test"]["input_ids"])

    train_tensor = {
        "train": ds_train
    }
    val_tensor = {
        "val": ds_val
    }

    save_file(train_tensor, train_path)

    save_file(val_tensor, val_path)


if __name__ == "__main__":
    app.run()
