import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import pickle 
    import marimo as mo
    from pathlib import Path

    from tokenizer import download, tokenize, tokenize_char, save_tensors
    from hyperparameters import intialize_hyperparameters
    from train import initialize_model, train_sequential_batches, save_checkpoint
    from sample import SampleConfig, sample


@app.cell
def _():
    ### Set Hyperpameters ###
    cpu_config = intialize_hyperparameters(
        batch_size = 8,
        block_size = 64,
        n_layer = 4,
        n_head = 4,
        n_embd = 64,
        dropout = 0,
        learning_rate = 3e-3,
        max_iters = 1000,
        epoch_iters = 0, # This overrides max_iters
        beta2 = 0.99,
        log_iters = 100
    )

    gpu_config = intialize_hyperparameters(
        batch_size = 64,
        block_size = 256,
        n_layer = 6,
        n_head = 6,
        n_embd = 384,
        dropout = 0.1,
        learning_rate = 1e-3,
        max_iters = 5000,
        epoch_iters = 3, # This overrides max_iters
        log_iters = 100
    )

    config = cpu_config
    config
    return (config,)


@app.cell
def _(config):
    ### Tokenize ###
    data_path = config["root_path"] / "data"
    tensor_path = data_path / "train.safetensors"

    data_path.mkdir(parents=True, exist_ok=True)

    if not tensor_path.exists():
        ds = download()
        ds_tok = tokenize_char(config, ds)
        save_tensors(config, ds_tok)

    ### Pre-Training ###
    checkpoint_path = data_path / "checkpoint.pt"

    if not checkpoint_path.exists():
        ### Initialize Model ###
        model, optimizer = initialize_model(config)

        ### Train Model ###
        losses, _ = train_sequential_batches(config, model, optimizer)
        save_checkpoint(config, model, optimizer, losses)
    return


@app.cell
def _():
    ### Sample ###
    sample_config = SampleConfig()
    sample_config

    sample(sample_config)
    return


if __name__ == "__main__":
    app.run()
