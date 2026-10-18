# C++ ML Playground

[![C++](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/cpp.yml/badge.svg?branch=master)](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/cpp.yml)
[![Server](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/server.yml/badge.svg?branch=master)](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/server.yml)
[![Client](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/client.yml/badge.svg?branch=master)](https://github.com/Nicolas2912/cpp-ml-react/actions/workflows/client.yml)

Train a linear regressor and a neural network, both written from scratch in C++, on the same data and compare them on a held-out test set. Training and validation loss stream live to a React UI over WebSockets, so you can watch a network overfit and then fix it.

![Loading the overfitting demo: the unregularized network's validation loss turns upward, early stopping restores the best epoch, and the network then beats linear regression on the held-out points](docs/demo.gif)

Across 100 freshly drawn noisy datasets, an unregularized 1-32-32-1 network memorizes the noise and does worse than a straight line (mean test MSE 1.13 vs 0.96). Weight decay brings it to 0.69 and validation-based early stopping to 0.75, against a noise floor of 0.52. See [Overfitting lab](#overfitting-lab).

**Highlights**

- **From-scratch C++ engine.** Backpropagation, mini-batch SGD and Adam, weight decay (L2 for SGD, decoupled AdamW for Adam), early stopping that restores the best weights, sigmoid/tanh/ReLU activations with Xavier/He initialization, closed-form least squares, and full-precision model serialization. No ML libraries.
- **Verified math.** Unit tests check backprop, including the weight-decay term, against finite differences, and check AdamW and mini-batch SGD updates against PyTorch to 1e-12 ([reference script](cpp/tests/reference/optimizer_reference.py)).
- **Honest evaluation.** Test points are held out from both models. The network also holds out validation points from its training data for early stopping, so the test set is never used for model selection. Scaling uses the network's training points only, and every MSE is reported in original Y units.
- **Live training.** Each run is an isolated job in its own C++ process, streaming training and validation loss over a WebSocket, with cancellation, timeouts, and per-session ownership.
- **Real inference.** New predictions reload the trained C++ weights rather than interpolating the plotted curve.
- **Tested end to end.** C++ unit tests, server integration tests that drive real engine processes, and UI tests all run in CI, along with the production build.

## Architecture

```mermaid
flowchart LR
    UI["React + Chart.js<br/>(Vite)"]
    API["Node.js / Express<br/>job queue"]
    ENG["C++ engine<br/>linear_regression_app"]
    UI -- "POST /api/jobs" --> API
    API -- "WebSocket /ws<br/>loss stream + results" --> UI
    API -- "spawn per job<br/>CSV over stdin" --> ENG
    ENG -- "key=value over stdout<br/>epoch=…,mse=…" --> API
```

The server validates and scales the data, splits it, and runs each training job as a bounded C++ subprocess. It parses streamed progress lines and forwards them only to the WebSocket session that owns the job. Trained models are kept in memory as serialized weights, so later predictions run the real network.

![Dark mode: the tanh network fitting the curve sample while linear regression stays flat](docs/dark-mode.png)

## Overfitting lab

Open the **Neural Network** tab and choose **Load overfitting demo**. It loads 40 noisy samples of 2 sin(x) and a 1-32-32-1 tanh network trained with full-batch Adam for 5,000 epochs. That network has far more capacity than 26 noisy points need. Train it and the curves split: training loss keeps falling while validation loss bottoms out near epoch 1,600 and then climbs as the network memorizes noise.

![Training and validation loss without regularization (left) and with early stopping, which ends at epoch 2,588 and restores epoch 1,588 (right)](docs/overfitting-curves.png)

Turn on **Early stopping** to end training once validation loss has not improved for the patience window, and to restore the weights from the best validation epoch. Alternatively, set **Weight decay** to shrink the weights at every step. On this sample, early stopping brings test MSE from 0.88 to 0.33 and weight decay to 0.34. That is about what the true function itself scores on these noisy points (0.33), and well ahead of linear regression's 0.67.

One sample is an anecdote, so [`server/scripts/overfitting-study.js`](server/scripts/overfitting-study.js) repeats the experiment on 100 independently drawn datasets. Each dataset uses the same generator with a different seed, and each run goes through the real job pipeline and C++ engine. The patience and decay settings were chosen on other seeds, so these datasets are fresh. Every regularized run is paired with the unregularized run on the same data:

| Model                                   | Mean test MSE | Change vs. no regularization (95% CI) | Better on |
| --------------------------------------- | ------------- | ------------------------------------- | --------- |
| Linear regression                       | 0.955         | −0.177 (−0.334 to −0.021)             | 58/100    |
| Network, no regularization              | 1.133         | —                                     | —         |
| Network, early stopping (patience 1000) | 0.752         | −0.381 (−0.529 to −0.232)             | 72/100    |
| Network, weight decay 1                 | 0.691         | −0.442 (−0.579 to −0.305)             | 78/100    |
| Noise floor (true function)             | 0.520         |                                       |           |

Each dataset has 26 points to learn from, 6 for validation, and 8 for testing. Early stopping ran 2,228 of 5,000 epochs on average. Reproduce with `node scripts/overfitting-study.js` in `server/` (about a minute). The figures above come from the Linux build used by CI and Docker; other platforms can differ slightly in floating point.

The optimizer matters here. In tuning runs with the original per-sample SGD at a fixed learning rate, the validation curve was too noisy for early stopping to pick a good epoch, and it did not beat the unregularized network on average. Full-batch Adam gives smooth curves on which validation-based model selection works.

## Current features

- Compose comma-separated X/Y data, load linear, curved, wave, or noisy-wave samples, or generate randomized data with point-count and linearity controls.
- Train linear regression and the neural network independently from their original model tabs, or train both together on the same deterministic 80/20 split.
- Inspect the raw dataset and the combined model overlay, including training points, held-out points, fitted curves, and new predictions.
- Compare training MSE, held-out test MSE, test R², and compute time in one table. All displayed MSE values use the original Y scale.
- Stream neural-network training and validation loss during training, mark the best validation epoch, and cancel an active run.
- Train the network with per-sample or mini-batch SGD or Adam, add weight decay, and enable early stopping with a patience window.
- Build a `1-N-…-1` neural architecture visually: add or remove hidden layers, choose 1–32 neurons per hidden layer, select nodes, and pan, zoom, or fit the graph.
- Choose the hidden-layer activation: sigmoid (default), tanh with Xavier initialization, or ReLU with He initialization.
- Predict from either original model panel or evaluate both trained models together. Neural predictions reload the trained C++ weights rather than interpolating displayed points.
- Switch between the original light and dark themes. The full workflow is responsive on narrow screens.

## Run with Docker

```sh
docker build -t cpp-ml-playground .
docker run --rm -p 8080:3001 cpp-ml-playground
```

Open http://localhost:8080. The image compiles and tests the C++ engine, builds the client, and runs the server as a non-root user. A single process serves the page, the API, and the WebSocket.

## Run locally

Requires Node.js 22.12+ (Node 24 LTS recommended), npm, make, and a C++11 compiler. GCC uses OpenMP; Clang can build sequentially when libomp is unavailable.

```sh
make -C cpp
npm --prefix server ci
npm --prefix client ci
```

Start these in separate terminals:

```sh
npm --prefix server start
npm --prefix client start
```

Open http://127.0.0.1:3000. Vite proxies `/api` and `/ws` to the API on port 3001. Both servers bind to loopback. Use the Vite origin for browser requests.

## What the comparison measures

- The server shuffles pairs with seed 42 and holds out 20% (rounded, at least one point) for datasets of 5 or more pairs. Both models use the same test points, including when trained separately. Datasets of 2–4 pairs remain supported as training-only runs; test metrics are undefined.
- When at least 10 training points remain (12 or more pairs), the network holds out a further 20% of them (at least two) as a validation set. Linear regression has no hyperparameters to tune, so it fits on all training points. Early stopping needs this validation set.
- NN normalization uses **the network's training values only**: not validation, not test. Training loss, validation loss, train MSE, and test MSE are all returned in original Y units squared.
- Lower test MSE means better predictions on this particular split. A low training error alone does not establish generalization. Repeated tuning against one test split can overfit that split.
- Test R² compares predictions with the test-set mean: 1 is perfect, 0 matches that baseline, and negative values are worse. It is undefined (`null`) for constant test targets or fewer than two test points.
- Compute time includes training and engine evaluation/output, excluding process startup and network latency. It is not a rigorous benchmark.
- NN hidden layers use the chosen activation (sigmoid by default, or tanh/ReLU); the output is linear. The loss is 0.5 · (prediction − target)², averaged over each mini-batch. SGD (batch size 1 by default) and Adam (β₁ 0.9, β₂ 0.999, ε 1e-8) use fixed initialization and shuffling seeds. Weight decay applies to weights, not biases. Repeated runs with the same build and inputs are reproducible; timings and cross-platform floating-point results may vary.
- Validation loss is measured after every epoch. Without early stopping, the final weights are the last epoch's, and the best validation epoch is only reported. With early stopping, training ends after `patience` epochs without improvement and the best epoch's weights are restored.
- “Predict both” runs C++ inference using the actual trained parameters. NN weights round-trip at full double precision. New predictions are neither interpolation nor endpoint clamping.

## Structure

| Location                              | Purpose                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `client/src/App.jsx`                  | Dataset, settings, and comparison workflow                              |
| `client/src/hooks/usePlayground.js`   | Original frontend workflows, independent model state, and themed charts |
| `client/src/NNVisualizer.jsx`         | Editable network blueprint, node selection, zoom, and pan               |
| `client/src/hooks/useComparison.js`   | Connection lifecycle and job updates                                    |
| `client/src/components/`              | Fit/loss charts, score table, prediction form                           |
| `server/server.js`                    | HTTP/WebSocket endpoints and session ownership                          |
| `server/jobs.js`                      | Training, evaluation, model retention, cancellation                     |
| `server/engine.js`                    | Bounded subprocess execution and streamed output parsing                |
| `server/data.js`                      | Validation, splitting, scaling, metrics                                 |
| `cpp/main_server.cpp`                 | CLI protocol                                                            |
| `cpp/neural_network.*`                | Network training, optimizers, early stopping, inference, serialization  |
| `cpp/tests/reference/`                | PyTorch script that generates the optimizer reference values            |
| `server/scripts/overfitting-study.js` | Regularization study across many noisy datasets                         |
| `cpp/linear_regression.*`             | Analytical regression and gradient-descent implementation               |

## Sessions and limits

A WebSocket connection receives an opaque session token. HTTP requests send it as `Authorization: Bearer <token>`. Jobs have unique IDs; only the owning session can read, cancel, or predict with them. Progress goes to that connection only.

Models live in server memory, with a 30-minute TTL and at most three retained runs per session. The retention logic keeps the latest independently trained LR and NN models available for comparison where possible. Disconnecting or restarting the server discards them; the UI reconnects and asks you to train again. This is a local playground, without accounts or disk model storage.

Limits: 2–1,000 input pairs; finite values within ±1,000,000; one input/output neuron; up to four hidden layers of 32 neurons; 1–10,000 epochs; learning rate in `(0, 1]`; weight decay in `[0, 1]`; batch size 1–1,000; patience 0–10,000 epochs. A parameter × point × epoch budget rejects overly large runs. Each engine process has a 30-second timeout and a 2 MB output limit. The server allows four simultaneous compute operations, one training job per session, 32 connections, and 64 retained jobs globally. Disconnect and cancellation terminate associated subprocesses.

API:

| Method / path                | Purpose                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GET /api/health`            | API liveness                                                                                                                                                                                                                                                                                                                                                                               |
| `POST /api/jobs`             | Train the selected model(s); body: `{x, y, layers, learningRate, epochs, activation, optimizer, batchSize, weightDecay, patience, model}` (`activation`: `sigmoid` (default), `tanh`, or `relu`; `optimizer`: `sgd` (default) or `adam`; `batchSize` default 1; `weightDecay` default 0; `patience` default 0 = no early stopping; `model`: `lr`, `nn`, or default `both`); returns `{id}` |
| `GET /api/jobs/:id`          | Recover the latest status, loss, and results                                                                                                                                                                                                                                                                                                                                               |
| `POST /api/jobs/:id/cancel`  | Cancel an active run                                                                                                                                                                                                                                                                                                                                                                       |
| `POST /api/jobs/:id/predict` | Evaluate both models; body: `{x, model}` (model optional)                                                                                                                                                                                                                                                                                                                                  |

WebSocket `/ws` events: `session`, `progress`, `completed`, `failed`, `cancelled`. Job events include `id`. The former `/api/lr_*` and `/api/nn_train_predict` HTTP routes have been replaced by this unified job API.

## Verify

```sh
OMP_NUM_THREADS=1 make -C cpp test_all
npm --prefix server test
npm --prefix client test
npm --prefix client run build
```

Server integration tests launch real C++ processes and HTTP/WebSocket clients. They cover the validation split and validation-loss stream, early stopping with best-weight restoration, weight decay, inference, metric scaling, repeatability, independent and paired training, small datasets, session isolation, model retention, validation, cancellation, expiry, capacity, malformed output, launch failure, and timeout. UI tests cover the overfitting demo and regularization controls, the restored composer and themes, separate model tabs, comparison training, prediction, architecture editing, node selection, cancellation, invalid data, stale updates, and disconnects. CI runs these checks and the production build.

Browser check: load the overfitting demo, train it, then retrain with early stopping and with weight decay, and check the loss curves, best-epoch marker, and validation points. Train LR and NN separately, then compare both; inspect held-out scores and predict through each original model input and the paired prediction form. In the Neural Network tab, show the blueprint, add/remove hidden layers, adjust neuron counts, select nodes, and use zoom/fit controls. Change the dataset, cancel a long run, and toggle the light/dark theme. Repeat at a narrow mobile width. Test loss of the API connection and recovery.

The production bundle is written to `client/build`. Set `STATIC_DIR=client/build` and `HOST=0.0.0.0` to have the Node server serve it on the same origin as `/api` and `/ws`; the Dockerfile does this.

## Network builder

Open the **Neural Network** tab to show the restored blueprint. The original layer-size input and the graph controls stay synchronized. Add up to four hidden layers, choose 1–32 neurons per hidden layer, remove layers, and select a node to inspect its layer. Input/output stay at one neuron for this X-to-Y regression app. Editing architecture invalidates the previous NN result while preserving a completed LR model for the same dataset. Model parameters are disabled during training.
