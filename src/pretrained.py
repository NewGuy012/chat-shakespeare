import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from transformers import pipeline, set_seed, GenerationConfig

    generator = pipeline("text-generation", model="gpt2")
    prompt = "Hello, I'm a language model,"

    set_seed(42)

    config = GenerationConfig(
        max_new_tokens = 50,
        do_sample = True,
        num_return_sequences = 3,
    )
    generated_txt = generator(prompt, generation_config=config)
    generated_txt
    return


@app.cell
def _():
    import torch
    from model import GPT
    from hyperparameters import intialize_hyperparameters
    from sample import SampleConfig, sample

    cpu_config = intialize_hyperparameters()
    model = GPT.from_pretrained("gpt2")
    sample_config = SampleConfig(
        load_meta = False,
        start = "Hello, I'm a language model,",
        num_samples = 3,
        max_new_tokens = 50,
        temperature = 1,
        top_k = 50)
    torch.manual_seed(42)
    sample(cpu_config, sample_config, model)
    return


if __name__ == "__main__":
    app.run()
