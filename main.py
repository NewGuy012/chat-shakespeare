import marimo

__generated_with = "0.26.0"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    from hyperparameters import TrainConfig
    from sample import SampleConfig, sample
    from tokenizer import download, save_tensors, tokenize_char, tokenize_gpt2
    from train import save_checkpoint, train


@app.cell
def _():
    # Hardware-aware defaults for cpu, mps, or cuda (override any field as needed)
    config = TrainConfig.auto()
    config
    return (config,)


@app.cell
def _(config):
    overwrite = False
    tokenize(config, overwrite=overwrite)
    checkpoint_dir = training(config, overwrite=overwrite)
    return (checkpoint_dir,)


@app.cell
def _(checkpoint_dir, config):
    sample_config = SampleConfig(
        num_samples=1,
        max_new_tokens=500,
    )
    sample(config, sample_config)
    return


@app.function
def tokenize(config, overwrite=False):
    ready = (
        config.train_path.exists()
        and config.val_path.exists()
        and config.meta_path.exists()
    )
    if overwrite or not ready:
        ds = download(config)
        tokenized = (
            tokenize_gpt2(config, ds)
            if config.init_from.startswith("gpt2")
            else tokenize_char(config, ds)
        )
        save_tensors(config, tokenized)


@app.function
def training(config, overwrite=False):
    ckpt_ready = (config.checkpoint_dir / "model.safetensors").exists() and (
        config.checkpoint_dir / "config.json"
    ).exists()
    if overwrite or not ckpt_ready:
        accelerator, model, losses = train(config)
        save_checkpoint(config, accelerator, model, losses)
    return config.checkpoint_dir


if __name__ == "__main__":
    app.run()
