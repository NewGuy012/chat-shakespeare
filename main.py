import marimo

__generated_with = "0.26.0"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    from hyperparameters import TrainConfig
    from sample import SampleConfig, sample
    from tokenizer import prepare_tokens
    from train import ensure_checkpoint, load_checkpoint


@app.cell
def _():
    # Hardware-aware defaults for cpu, mps, or cuda (override any field as needed)
    config = TrainConfig.auto()
    config
    return (config,)


@app.cell
def _(config):
    overwrite_tokens = False
    token_dir = prepare_tokens(config, overwrite=overwrite_tokens)
    return (token_dir,)


@app.cell
def _(config, token_dir):
    _ = token_dir
    overwrite_training = False
    checkpoint_dir = ensure_checkpoint(config, overwrite=overwrite_training)
    return (checkpoint_dir,)


@app.cell
def _(checkpoint_dir, config):
    _, model = load_checkpoint(checkpoint_dir, device=config.device)
    return (model,)


@app.cell
def _(config, model):
    sample_config = SampleConfig(
        num_samples=1,
        max_new_tokens=500,
    )
    sample(config, sample_config, model=model)
    return


if __name__ == "__main__":
    app.run()
