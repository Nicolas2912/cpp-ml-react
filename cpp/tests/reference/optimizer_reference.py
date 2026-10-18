"""Reference values for the optimizer tests in neural_network_tests.cpp.

Builds the engine's 1-3-2-1 tanh test network in PyTorch, takes three mini-batch steps with
AdamW and with SGD plus an L2 penalty, and prints the resulting parameters as C++ initializers.
Regenerate with:  uv run --no-project --python 3.12 --with torch --with numpy python cpp/tests/reference/optimizer_reference.py
"""

import torch

torch.set_default_dtype(torch.float64)

WEIGHTS = [
    [[0.4], [-0.3], [0.8]],
    [[0.5, -0.6, 0.7], [-0.2, 0.3, 0.9]],
    [[0.6, -0.4]],
]
BIASES = [[0.1, 0.2, -0.05], [0.05, -0.1], [0.02]]
BATCHES = [
    ([0.7, -0.2, 0.4], [0.2, -0.5, 0.1]),
    ([0.1, 0.9], [0.6, 0.3]),
    ([-0.8, 0.3, 0.5], [-0.4, 0.0, 0.7]),
]
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.1


def train(make_optimizer):
    weights = [torch.tensor(w, requires_grad=True) for w in WEIGHTS]
    biases = [torch.tensor(b, requires_grad=True) for b in BIASES]
    # Weight decay applies to weights only, as in the engine.
    optimizer = make_optimizer(
        [{"params": weights, "weight_decay": WEIGHT_DECAY}, {"params": biases, "weight_decay": 0.0}]
    )
    for xs, ys in BATCHES:
        optimizer.zero_grad()
        a = torch.tensor(xs).unsqueeze(1)
        for i, (w, b) in enumerate(zip(weights, biases)):
            a = a @ w.T + b
            if i < len(weights) - 1:
                a = torch.tanh(a)
        # The engine's per-sample loss is 0.5 * (prediction - target)^2, averaged over the batch.
        loss = 0.5 * ((a.squeeze(1) - torch.tensor(ys)) ** 2).mean()
        loss.backward()
        optimizer.step()
    return [p.detach().flatten().tolist() for pair in zip(weights, biases) for p in pair]


def cpp(name, values):
    flat = ", ".join(f"{v:.17g}" for layer in values for v in layer)
    print(f"    const std::vector<double> {name}{{{flat}}};")


cpp("expected_adamw", train(lambda groups: torch.optim.AdamW(groups, lr=LEARNING_RATE)))
cpp("expected_sgd", train(lambda groups: torch.optim.SGD(groups, lr=LEARNING_RATE)))
