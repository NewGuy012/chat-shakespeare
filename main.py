import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import marimo as mo
    from pathlib import Path

    from tokenizer import download, tokenize
    from hyperparameters import intialize_hyperparameters
    from train import initialize_model, train_sequential_batches
    from sample import SampleConfig, sample

    from accelerate import Accelerator


@app.cell
def _(save):
    ### Tokenize ###
    file_name = "train-validation.safetensors"
    root_file = Path(__file__).parent
    save_path = root_file / "data" / file_name

    if not save_path.exists():
        ds = download()
        ds_tok = tokenize(ds)
        save(ds, save_path)
    return


@app.cell
def _():
    ### Hyperpameters ###
    hyper_config = intialize_hyperparameters(
        batch_size = 64,
        block_size = 256,
        n_layer = 6,
        n_head = 6,
        n_embd = 384,
        dropout = 0.1,
        learning_rate = 1e-3,
        max_iters = 5000,
        epoch_iters = 1 # This overrides max_iters
    )
    return (hyper_config,)


@app.cell
def _(hyper_config):
    ### Model ###
    model, optimizer = initialize_model(hyper_config)
    return model, optimizer


@app.cell
def _(hyper_config, model, optimizer):
    ### Train ###
    train_sequential_batches(hyper_config, model, optimizer)
    return


@app.cell
def _(hyper_config, model):
    ### Sample ###
    sample_config = SampleConfig()
    print(sample_config)
    sample(hyper_config, sample_config, model)
    return


if __name__ == "__main__":
    app.run()
