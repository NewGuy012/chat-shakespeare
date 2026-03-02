import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from accelerate import Accelerator

    # device = 'cpu'
    accelerator = Accelerator()

    # model = torch.nn.Transformer().to(device)
    model = torch.nn.Transformer()

    optimizer = torch.optim.Adam(model.parameters())

    dataset = load_dataset('jchwenger/tiny_shakespeare')
    data = torch.utils.data.DataLoader(dataset, shuffle=True)

    model, optimizer, data = accelerator.prepare(model, optimizer, data)

    model.train()
    for epoch in range(10):
        for source, targets in data:
            # source = source.to(device)
            # targets = targets.to(device)

            optimizer.zero_grad()

            output = model(source)
            loss = F.cross_entropy(output, targets)

            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()
    return


if __name__ == "__main__":
    app.run()
