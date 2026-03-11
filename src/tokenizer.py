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
    from datasets import DatasetDict, load_dataset


@app.function
def download():
    # jchwenger/tiny_shakespeare 
    dataset = load_dataset("Trelis/tiny-shakespeare")
    dataset = dataset.rename_column("Text", "text")

    return dataset


@app.function
def tokenize(ds):
    enc = tiktoken.get_encoding("gpt2")

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

    return tokenized


@app.function
def tokenize_tiktoken(ds):
    enc = tiktoken.get_encoding("gpt2")

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

    return tokenized


@app.function
def tokenize_char(config, ds):
    root_path = config["root_path"]
    tensor_path = root_path / "data" / "train.safetensors"

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
    stoi_dict = { ch:i for i,ch in enumerate(chars) }
    itos_dict = { i:ch for i,ch in enumerate(chars) }

    # encode both to integers
    ds_train = encode_char(stoi_dict, ds_train)
    ds_val = encode_char(stoi_dict, ds_val)

    ds = {
        "train": {"input_ids": ds_train},
        "test": {"input_ids": ds_val}
    }

    meta = {
        'vocab_size': vocab_size,
        'itos': itos_dict,
        'stoi': stoi_dict,
    }
    save_meta(config, meta)

    return ds


@app.function
def encode_char(stoi_dict, s):
    # encoder: take a string, output a list of integers
    return [stoi_dict[c] for c in s]


@app.function
def decode_char(itos_dict, l):
    # decoder: take a list of integers, output a string
    return ''.join([itos_dict[i] for i in l])


@app.function
def save_meta(config, meta):
    root_path = config["root_path"]

    save_path = root_path / "data" / "meta.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(meta, f)


@app.function
def save_tensors(config, ds):
    root_path = config["root_path"]
    train_path = root_path / "data" / "train.safetensors"
    val_path = root_path / "data" / "val.safetensors"

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
