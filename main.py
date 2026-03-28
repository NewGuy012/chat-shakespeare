import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import pickle 
    import marimo as mo
    from pathlib import Path

    from hyperparameters import intialize_hyperparameters
    from tokenizer import download, tokenize_gpt2, tokenize_char, save_tensors
    from train import initialize_model, train_sequential_batches, save_checkpoint
    from sample import SampleConfig, sample


@app.cell
def _():
    ### Set Hyperpameters ###
    cpu_config = intialize_hyperparameters(
        batch_size = 32,
        block_size = 256,
        n_layer = 4,
        n_head = 4,
        n_embd = 256,
        dropout = 0,
        learning_rate = 2e-3,
        max_iters = 5000,
        epoch_iters = 0, # This overrides max_iters
        beta2 = 0.99,
        log_iters = 500
    )

    gpu_config = intialize_hyperparameters(
        batch_size = 64,
        block_size = 256,
        n_layer = 6,
        n_head = 6,
        n_embd = 384,
        dropout = 0.1,
        learning_rate = 1e-3,
        max_iters = 2000,
        epoch_iters = 0, # This overrides max_iters
        log_iters = 100
    )

    config = cpu_config
    config
    return (config,)


@app.cell
def _(config):
    ### Tokenize ###
    bOverwrite = True
    tokenize(config, bOverwrite)

    ### Training ###
    bOverwrite = True
    training(config, bOverwrite)
    return (bOverwrite,)


@app.cell
def _(bOverwrite, config):
    ### Sample ###
    sample_config = SampleConfig(
        bOverwrite = bOverwrite,
        num_samples = 1,
        max_new_tokens = 500
    )
    sample_config

    sample(config, sample_config)
    return


@app.function
def tokenize(config, bOverwrite):
    ds = download(config)

    tensor_path = config["tensor_path"]

    if bOverwrite or not tensor_path.exists():
        ds_tok = tokenize_char(config, ds)
        save_tensors(config, ds_tok)


@app.function
def training(config, bOverwrite):
    checkpoint_path = config["checkpoint_path"]

    if bOverwrite or not checkpoint_path.exists():
        # Initialize
        model, optimizer = initialize_model(config)

        # Train
        losses, _ = train_sequential_batches(config, model, optimizer)
        save_checkpoint(config, model, optimizer, losses)


if __name__ == "__main__":
    app.run()
